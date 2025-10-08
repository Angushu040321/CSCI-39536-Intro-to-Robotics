import csv, os, math
import matplotlib.pyplot as plt


def _to_float_or_nan(s):
    s = s.strip() if isinstance(s, str) else s
    return float(s) if s not in (None, "",) else float('nan')


def _any_finite(xs):
    return any(math.isfinite(x) for x in xs)


def main():
    path = os.path.expanduser('~/ros2_ws/ekf_compare.csv')
    t = []
    odom_x, odom_y = [], []
    filt_x, filt_y = [], []
    gt_x, gt_y = [], []

    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row['t']))
            odom_x.append(float(row['odom_x']))
            odom_y.append(float(row['odom_y']))
            filt_x.append(float(row['filt_x']))
            filt_y.append(float(row['filt_y']))
            gt_x.append(_to_float_or_nan(row.get('gt_x', "")))
            gt_y.append(_to_float_or_nan(row.get('gt_y', "")))

    plt.figure()
    plt.plot(odom_x, odom_y, label='Odometry (/odom)')
    plt.plot(filt_x, filt_y, label='EKF (/odometry/filtered)')
    if _any_finite(gt_x) and _any_finite(gt_y):
        plt.plot(gt_x, gt_y, label='Ground truth', linewidth=3, alpha=0.5)

    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Trajectory Comparison')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
