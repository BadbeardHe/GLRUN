import numpy as np
from kinect_init import *
from scipy import ndimage

PI = 3.14159265358979323846


def processPixelStage2(m):
    """
        Tackling 3d tof artifacts through learning and the flat dataset
        Guo, Qi, Iuri Frosio, Orazio Gallo, Todd Zickler, and Jan Kautz
        European Conference on Computer Vision (ECCV), 2018
        Project Page : https://research.nvidia.com/publication/2018-09_tackling-3d-tof-artifacts-through-learning-and-flat-dataset
    """
    # m is (424, 512, 9)
    # the first three is measurement a
    # the second three is measurement b
    # the third three is amplitude
    tmp0 = np.arctan2(m[:, :, 3:6], m[:, :, 0:3])
    flg = (tmp0 < 0.0).astype(np.float32)
    tmp0 = flg * (tmp0 + PI * 2) + (1 - flg) * tmp0

    tmp1 = np.sqrt(m[:, :, 0:3] ** 2 + m[:, :, 3:6] ** 2) * prms['ab_multiplier']

    ir_sum = np.sum(tmp1, axis=-1)

    # disable disambiguation
    ir_min = np.min(tmp1, axis=-1)

    # phase mask
    phase_msk1 = (ir_min > prms['individual_ab_threshold']).astype(np.float32)
    phase_msk2 = (ir_sum > prms['ab_threshold']).astype(np.float32)
    phase_msk_t = phase_msk1 * phase_msk2

    # compute phase
    t0 = tmp0[:, :, 0] / (2.0 * PI) * 3.0
    t1 = tmp0[:, :, 1] / (2.0 * PI) * 15.0
    t2 = tmp0[:, :, 2] / (2.0 * PI) * 2.0

    t5 = np.floor((t1 - t0) * 0.3333333 + 0.5) * 3.0 + t0
    t3 = t5 - t2
    t4 = t3 * 2.0

    c1 = (t4 > -t4).astype(np.float32)
    f1 = c1 * 2.0 + (1 - c1) * (-2.0)
    f2 = c1 * 0.5 + (1 - c1) * (-0.5)
    t3 = t3 * f2
    t3 = (t3 - np.floor(t3)) * f1

    c2 = ((np.abs(t3) > 0.5) & (np.abs(t3) < 1.5)).astype(np.float32)
    t6 = c2 * (t5 + 15.0) + (1 - c2) * t5
    t7 = c2 * (t1 + 15.0) + (1 - c2) * t1
    t8 = (np.floor((t6 - t2) * 0.5 + 0.5) * 2.0 + t2) * 0.5

    t6 /= 3.0
    t7 /= 15.0

    # transformed phase measurements
    t9 = t8 + t6 + t7
    t10 = t9 / 3.0  # some avg

    t6 = t6 * 2.0 * PI
    t7 = t7 * 2.0 * PI
    t8 = t8 * 2.0 * PI

    t8_new = t7 * 0.826977 - t8 * 0.110264
    t6_new = t8 * 0.551318 - t6 * 0.826977
    t7_new = t6 * 0.110264 - t7 * 0.551318

    t8 = t8_new
    t6 = t6_new
    t7 = t7_new

    norm = t8 ** 2 + t6 ** 2 + t7 ** 2
    mask = (t9 > 0.0).astype(np.float32)
    t10 = t10 * mask

    slope_positive = float(prms['ab_confidence_slope'] > 0)

    ir_min_ = np.min(tmp1, axis=-1)
    ir_max_ = np.max(tmp1, axis=-1)

    ir_x = slope_positive * ir_min_ + (1 - slope_positive) * ir_max_

    ir_x = np.log(ir_x)
    ir_x = (ir_x * prms['ab_confidence_slope'] * 0.301030 + prms['ab_confidence_offset']) * 3.321928
    ir_x = np.exp(ir_x)
    ir_x = np.maximum(prms['min_dealias_confidence'], ir_x)
    ir_x = np.minimum(prms['max_dealias_confidence'], ir_x)
    ir_x = ir_x ** 2

    mask2 = (ir_x > norm).astype(np.float32)

    t11 = t10 * mask2

    mask3 = (prms['max_dealias_confidence'] ** 2 > norm).astype(np.float32)
    t10 = t10 * mask3
    phase = t11

    # mask out dim regions
    phase = phase * phase_msk_t

    # phase to depth mapping
    zmultiplier = z_table
    xmultiplier = x_table

    phase_msk = (phase > 0.0).astype(np.float32)
    phase = phase_msk * (phase + prms['phase_offset']) + (1 - phase_msk) * phase

    depth_linear = zmultiplier * phase
    max_depth = phase * prms['unambiguous_dist'] * 2

    cond1 = ((depth_linear > 0.0) & (max_depth > 0.0)).astype(np.float32)

    xmultiplier = (xmultiplier * 90) / (max_depth ** 2 * 8192.0)

    depth_fit = depth_linear / (-depth_linear * xmultiplier + 1)

    depth_fit = np.maximum(depth_fit, 0.0)
    depth = cond1 * depth_fit + (1 - cond1) * depth_linear

    return depth


def distance_transform(depth, mask):
    depth = np.array(depth, dtype=float)
    mask = np.array(mask, dtype=bool)
    
    if depth.shape != mask.shape:
        raise ValueError
    
    out_depth = depth.copy()
    
    known_values = depth[~mask]
    
    if len(known_values) == 0:
        return depth
    
    _, indices = ndimage.distance_transform_edt(
        mask,
        return_indices=True  
    )

    out_depth[mask] = depth[indices[0][mask], indices[1][mask]]
    
    return out_depth.astype(np.float32)


def iq2depth(concat_IQ, raw):
    order = [2, 5, 8]
    concat_IQ = np.transpose(concat_IQ, (1, 2, 0))  # (424, 512, 6)
    x = np.stack((
        concat_IQ[:, :, 3],
        concat_IQ[:, :, 1],
        concat_IQ[:, :, 5],
        concat_IQ[:, :, 2],
        concat_IQ[:, :, 0],
        concat_IQ[:, :, 4]), axis=2)
    
    for i in order:
        if i in [2, 5, 8]:
            amplitude = raw[:, :, i:i+1]
            x = np.concatenate((x, amplitude), axis=2)
    
    depth = processPixelStage2(x)
    depth = depth / 1e3
    depth = np.nan_to_num(depth, 0)
    mask = ((depth < 0.1) | (depth > 10)).astype(np.uint8)

    return distance_transform(depth, mask)



