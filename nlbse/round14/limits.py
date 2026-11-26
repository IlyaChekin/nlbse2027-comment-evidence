"""Calculate how far visible-comment repairs can move the existing local score."""

import json
import math

from audit import HERE, ROOT, write_json


def main():
    audit = json.loads((HERE/"audit_public.json").read_text())["public"]
    ref = json.loads((ROOT/"round13/limits.json").read_text())
    with_comments = audit["false_negative_scope"]["comment_without_base_tag"]
    no_comment = audit["false_negative_scope"]["no_parsed_comment"]
    mat = ref["mat_observed"]
    if mat["fn"] != with_comments + no_comment or mat["tp"] != audit["tp"] or mat["fp"] != audit["fp"]:
        raise ValueError("MAT accounting differs from the published reference")
    target = ref["required_mat_f1_if_fixed_vulnerability"]
    target_tp = math.ceil((mat["tp"]+mat["fn"]+mat["fp"])*target/(2-target))
    true_positive = mat["tp"]+with_comments
    f1 = 2*true_positive/(true_positive+mat["tp"]+mat["fn"]+mat["fp"])
    score_at_best_visible = .97 + .3*(f1-target)
    output = dict(observed_mat=mat, reference_assumptions="vulnerability decisions, forward time and GFLOPs fixed",
                  visible_comment_false_negatives=with_comments, no_parsed_comment_false_negatives=no_comment,
                  mat_f1_if_every_visible_comment_miss_recovered=f1,
                  local_score_if_every_visible_comment_miss_recovered=score_at_best_visible,
                  needed_mat_recoveries_at_same_fp=target_tp-mat["tp"],
                  needed_without_parsed_comments_after_every_visible_repair=max(0,target_tp-true_positive))
    write_json(HERE/"limits.json",output)
    print(output,flush=True)


if __name__=="__main__":
    main()
