import os
import fire
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result


def _read_imageset_file(path):
    with open(os.path.expanduser(path), "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate_once(
    result_path,
    gt_annos,
    dt_annos,
    min_depth,
    max_depth,
    current_class=0,
    coco=False,
    suffix="",
):
    ress = ""
    if current_class == -1:
        for i in range(5):
            if coco:
                res, _ = get_coco_eval_result(gt_annos, dt_annos, i)
                ress += res
            else:
                res, _ = get_official_eval_result(
                    gt_annos, dt_annos, i, depth_limit=(min_depth, max_depth)
                )
                ress += res
        current_class = "all"
    else:
        if coco:
            res, res_dict = get_coco_eval_result(gt_annos, dt_annos, current_class)
            ress += res
        else:
            res, res_dict = get_official_eval_result(
                gt_annos, dt_annos, current_class, depth_limit=(min_depth, max_depth)
            )
            ress += res

    output_folder = "/".join(result_path.split("/")[:-1])
    out_name = os.path.join(
        output_folder,
        f"results_class_{current_class}_{min_depth}_{max_depth}" + suffix + ".txt",
    )
    print("-" * 100)
    print(ress)
    with open(out_name, "w") as f:
        f.write(ress)


DEPTH_EVAL = [(0, float("Inf")), (0, 10), (10, 20), (20, 30)]


def evaluate(
    label_path,
    result_path,
    label_split_file,
    current_class=0,
    coco=False,
    score_thresh=-1,
    min_depth=None,
    max_depth=None,
    suffix="",
):
    if min_depth is None:
        min_depth = 0
    if max_depth is None:
        max_depth = float("Inf")

    result_path = os.path.expanduser(result_path)
    label_path = os.path.expanduser(label_path)
    
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)

    if min_depth == max_depth == -1:
        for min_depth, max_depth in DEPTH_EVAL:
            evaluate_once(
                result_path,
                gt_annos,
                dt_annos,
                min_depth,
                max_depth,
                current_class=current_class,
                coco=coco,
                suffix=suffix,
            )
    else:
        evaluate_once(
            result_path,
            gt_annos,
            dt_annos,
            min_depth,
            max_depth,
            current_class=current_class,
            coco=coco,
            suffix=suffix,
        )


if __name__ == "__main__":
    fire.Fire()
