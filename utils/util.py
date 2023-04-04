import time

import cv2
import numpy
import scipy
import torch
import torchvision

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    import lap


def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def scale(coords, shape1, shape2, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, shape2[1])  # x1
    coords[:, 1].clamp_(0, shape2[0])  # y1
    coords[:, 2].clamp_(0, shape2[1])  # x2
    coords[:, 3].clamp_(0, shape2[0])  # y2
    return coords


def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(1.0, input_size / shape[0], input_size / shape[1])

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def non_max_suppression(outputs, conf_threshold=0.25, iou_threshold=0.45):
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 300  # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    start = time.time()
    output = [torch.zeros((0, 6), device=outputs.device)] * outputs.shape[0]
    for index, x in enumerate(outputs):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        x = x[(x[:, 5:6] == torch.tensor(0, device=x.device)).any(1)]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        output[index] = x[i]
        if (time.time() - start) > 0.5 + 0.05 * outputs.shape[0]:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * outputs.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return output


def merge_matches(m1, m2, shape):
    o, p, q = shape
    m1 = numpy.asarray(m1)
    m2 = numpy.asarray(m2)

    m1 = scipy.sparse.coo_matrix((numpy.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(o, p))
    m2 = scipy.sparse.coo_matrix((numpy.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(p, q))

    mask = m1 * m2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_o = tuple(set(range(o)) - {i for i, j in match})
    unmatched_q = tuple(set(range(q)) - {j for i, j in match})

    return match, unmatched_o, unmatched_q


def linear_assignment(cost_matrix, thresh, use_lap=True):
    # Linear assignment implementations with scipy and lap.lapjv
    if cost_matrix.size == 0:
        matches = numpy.empty((0, 2), dtype=int)
        unmatched_a = tuple(range(cost_matrix.shape[0]))
        unmatched_b = tuple(range(cost_matrix.shape[1]))
        return matches, unmatched_a, unmatched_b

    if use_lap:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = numpy.where(x < 0)[0]
        unmatched_b = numpy.where(y < 0)[0]
    else:
        # Scipy linear sum assignment is NOT working correctly, DO NOT USE
        y, x = scipy.optimize.linear_sum_assignment(cost_matrix)  # row y, col x
        matches = numpy.asarray([[i, x] for i, x in enumerate(x) if cost_matrix[i, x] <= thresh])
        unmatched = numpy.ones(cost_matrix.shape)
        for i, xi in matches:
            unmatched[i, xi] = 0.0
        unmatched_a = numpy.where(unmatched.all(1))[0]
        unmatched_b = numpy.where(unmatched.all(0))[0]

    return matches, unmatched_a, unmatched_b


def compute_iou(a_boxes, b_boxes):
    """
    Compute cost based on IoU
    :type a_boxes: list[tlbr] | np.ndarray
    :type b_boxes: list[tlbr] | np.ndarray

    :rtype iou | np.ndarray
    """
    iou = numpy.zeros((len(a_boxes), len(b_boxes)), dtype=numpy.float32)
    if iou.size == 0:
        return iou
    a_boxes = numpy.ascontiguousarray(a_boxes, dtype=numpy.float32)
    b_boxes = numpy.ascontiguousarray(b_boxes, dtype=numpy.float32)
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = a_boxes.T
    b2_x1, b2_y1, b2_x2, b2_y2 = b_boxes.T

    # Intersection area
    inter_area = (numpy.minimum(b1_x2[:, None], b2_x2) - numpy.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (numpy.minimum(b1_y2[:, None], b2_y2) - numpy.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (box2_area + box1_area[:, None] - inter_area + 1E-7)


def iou_distance(a_tracks, b_tracks):
    """
    Compute cost based on IoU
    :type a_tracks: list[STrack]
    :type b_tracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(a_tracks) > 0 and isinstance(a_tracks[0], numpy.ndarray)) \
            or (len(b_tracks) > 0 and isinstance(b_tracks[0], numpy.ndarray)):
        a_boxes = a_tracks
        b_boxes = b_tracks
    else:
        a_boxes = [track.tlbr for track in a_tracks]
        b_boxes = [track.tlbr for track in b_tracks]
    return 1 - compute_iou(a_boxes, b_boxes)  # cost matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = numpy.array([det.score for det in detections])
    det_scores = numpy.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost


class KalmanFilterXYAH:
    """
    A Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = numpy.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = numpy.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = numpy.zeros_like(mean_pos)
        mean = numpy.r_[mean_pos, mean_vel]

        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        covariance = numpy.diag(numpy.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = numpy.diag(numpy.square(numpy.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_mat, mean)
        mean = numpy.dot(mean, self._motion_mat.T)
        covariance = numpy.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = numpy.diag(numpy.square(std))

        mean = numpy.dot(self._update_mat, mean)
        covariance = numpy.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrix of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [self._std_weight_position * mean[:, 3],
                   self._std_weight_position * mean[:, 3],
                   1e-2 * numpy.ones_like(mean[:, 3]),
                   self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 3],
                   self._std_weight_velocity * mean[:, 3],
                   1e-5 * numpy.ones_like(mean[:, 3]),
                   self._std_weight_velocity * mean[:, 3]]
        sqr = numpy.square(numpy.r_[std_pos, std_vel]).T

        motion_cov = [numpy.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = numpy.asarray(motion_cov)

        mean = numpy.dot(mean, self._motion_mat.T)
        left = numpy.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = numpy.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             numpy.dot(covariance, self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + numpy.dot(innovation, kalman_gain.T)
        new_covariance = covariance - numpy.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        metric : str
            Distance metric.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return numpy.sum(d * d, axis=1)
        elif metric == 'maha':
            factor = numpy.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return numpy.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError('invalid distance metric')
