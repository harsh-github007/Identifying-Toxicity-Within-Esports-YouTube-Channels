# Labelling guide

You label each comment in `to_label.csv` as toxic (`1`) or not toxic (`0`). The guide exists so the labels mean the same thing from the first row to the last, and so someone else could repeat them.

## The question

**Would this comment make a reasonable viewer or the creator feel attacked, degraded, or unsafe?** If yes, it is toxic.

Judge the words on the page, not a guess about the writer's mood. Read Hinglish in its everyday sense.

## Toxic (`1`)

| Category | What counts | Examples |
| --- | --- | --- |
| `insult` | Demeaning a person or group: calling them stupid, worthless, a noob as an attack | "tu kitna bada noob hai, uninstall kar", "you are trash bro" |
| `profanity` | Obscene or sexual abuse, including abbreviations and masked spellings | "bsdk", "ch*tiya", "mc bc" aimed at someone |
| `threat` | Wishing or threatening harm, or telling someone to hurt themselves | "mar ja", "kys" |
| `hate` | Attacks on religion, caste, region, gender or sexuality | Any slur, or "X log aise hi hote hai" used as an insult |

Fill the optional `category` column with the best fit. If more than one applies, choose the most severe: threat, then hate, then profanity, then insult.

## Not toxic (`0`)

- Gamer slang that praises: "op", "pro", "khatarnak gameplay", "bhai ne pel diya".
- Criticism of the game, stream or video without attacking a person: "stream lag kar raha hai", "ye update bakwas hai".
- Words that look rude but aren't in context: "chod" meaning leave ("piche chod diya"), "sala" as friendly filler.
- Spam, requests and self-promotion: "sub my channel", "bhai gun skin do", "UID 12345". These are annoying but not toxic.
- Swearing at no one in particular, when mild: "wtf that shot". Use `notes` if you are unsure.

## Hard cases

- **Banter between friends.** Label what an outside viewer would read. Heavy abuse is `1` even with a 😂.
- **Unclear meaning** (typos, another language you can't read). Leave `toxic` blank and write `unclear` in `notes`. Blank rows are skipped, not counted as clean.
- **Quoting someone else's abuse to complain about it.** `0`: the comment itself isn't attacking anyone.

## Good practice

- Label in two or three sittings rather than all 400 at once; attention drifts.
- After finishing, re-label 40 random rows without looking at your first answers. The share you agree with yourself (and ideally with a friend who labels the same 40) is worth reporting next to the results.

## Complete the September pilot

The saved sample key has 400 rows. The published label sheet contains 60 completed rows, leaving 340 to review. Keep the original sample IDs, strata and weights; do not redraw an easier sample or label only flagged comments.

Use the original local `to_label.csv` containing comment text. The repository's sample key alone is insufficient for annotation, and does not contain the missing text. Review the remaining rows using the rules above, retain uncertain rows as blank, and save completed labels as `data/annotation/labels.csv`. Have a second reviewer independently label a subset before treating results as validated.

Then rerun `python -m toxicity evaluate` against the original scored comments and sample key. Report the actual completed count, toxic count, uncertainty and reviewer agreement. Model-assisted draft labels must be distinguished from human-reviewed labels; neither missing text nor missing judgement may be filled by assumption.
