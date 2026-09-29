# Toxicity in Indian Esports YouTube Comments

How toxic are the comment sections of India's biggest gaming channels, and how well can automatic tools tell? This project collects recent comments from five channels, labels a sample by hand, measures three detection methods against those labels, and estimates the share of toxic comments per channel with confidence intervals.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harsh-github007/Identifying-Toxicity-Within-Esports-YouTube-Channels/blob/main/notebooks/run_study.ipynb)

> **Results:** see [`results/results.md`](results/results.md) once the study has been run. The pipeline and tests are complete; the numbers are produced by running the notebook.

## Why it's hard

Most comments are **Hinglish**: Hindi written in Latin script, mixed with English, with creative spelling ("bhaii", "opp", "ch*tiya"). Off-the-shelf toxicity models are trained mostly on English and European languages, and English sentiment tools read Hinglish as neutral. Any claim about toxicity here is only as good as the check against human judgement, so that check is built into the design.

## Method

1. **Collect** (`toxicity collect`). The YouTube Data API v3 fetches comments and replies from each channel's 10 most recent uploads. Author IDs are hashed, so no usernames are stored. Collection stops if two channels share videos or more than 15% of their comments, which guards against the data mix-up described below.
2. **Score** (`toxicity score`). Every comment gets two automatic judgements:
   - [Detoxify](https://github.com/unitaryai/detoxify)'s multilingual XLM-RoBERTa model, a toxicity probability from 0 to 1.
   - A transparent Hinglish and English [word list](config/lexicon.csv) with whole-word matching that sees through leetspeak, stretched letters and masked spellings.
3. **Sample** (`toxicity sample`). 400 comments are drawn for hand labelling, stratified by channel and over-sampling comments either method flagged, because toxic comments are rare. Each sampled comment carries a weight so all later estimates remain unbiased. The labelling sheet hides the channel and the automatic scores.
4. **Label** by hand, following the [labelling guide](docs/labelling-guide.md).
5. **Evaluate** (`toxicity evaluate`). Three methods are compared against the labels with 5-fold cross-validation, weighted precision, recall and F1, and bootstrap confidence intervals:
   - the word list,
   - Detoxify with its threshold tuned inside each fold,
   - a character n-gram logistic regression trained on the labels, which suits Hinglish spelling variation.
6. **Estimate** toxicity per channel two ways:
   - **Hand-labelled estimate:** the weighted share of labelled comments that are toxic, with a stratified Jeffreys interval that stays honest when toxic comments are rare.
   - **All-comments estimate:** the best method applied to every comment, corrected for its measured sensitivity and specificity with the Rogan–Gladen estimator. A bootstrap resamples both the labels and whole videos, since comments on one video aren't independent.

## Running it

**In Colab (recommended):** click the badge above. It needs a free YouTube Data API key: in [Google Cloud Console](https://console.cloud.google.com/), create a project, enable *YouTube Data API v3*, then go to *Credentials → Create credentials → API key*. The default settings use a few hundred of the 10,000 free daily quota units.

**Locally** (Python 3.10+):

```bash
pip install -e ".[model,dev]"
export YOUTUBE_API_KEY=...        # never commit this
python -m toxicity collect
python -m toxicity score
python -m toxicity sample -n 400  # then label data/annotation/to_label.csv, save as labels.csv
python -m toxicity evaluate       # writes results/results.md and charts
pytest                            # 19 tests, no network needed
```

Channels and collection limits are set in [`config/channels.yaml`](config/channels.yaml).

## Project layout

```
config/channels.yaml        channels and collection limits
config/lexicon.csv          word list, with categories and notes on excluded terms
src/toxicity/collect.py     YouTube API client, pagination, overlap guard
src/toxicity/text.py        cleaning and normalisation for Hinglish
src/toxicity/lexicon.py     word-list matching
src/toxicity/score.py       Detoxify scoring
src/toxicity/sample.py      stratified, weighted labelling sample
src/toxicity/evaluate.py    cross-validated metrics, prevalence estimates, intervals
src/toxicity/report.py      charts and results.md
notebooks/run_study.ipynb   the whole study in Colab
docs/labelling-guide.md     labelling rules
tests/                      pytest suite with a mocked YouTube API
```

## Changes from the 2021 version

The first version of this project (SRM Institute of Science and Technology) reported a 23% average toxicity rate across five channels. A 2026 review found that result could not stand:

- **The dataset covered one video, not six channels.** All six channel files held the same ~1,500 comments from a single Total Gaming stream, re-fetched at slightly different times and saved under different channel names.
- **The labels measured sentiment, not toxicity.** Comments were labelled with TextBlob's English sentiment score, which rates almost all Hinglish as neutral, so friendly comments were counted as toxic. The classifier then learned to reproduce TextBlob, which is why its accuracy looked high.

Running the new word list over that 2021 stream flags about 0.3% of comments, against the 23% originally reported. The project was rebuilt with fresh collection, human labels, and validated measurement. The original notebook and data remain in the git history.

## Limitations

- One annotator. A second annotator labelling a subset would allow inter-annotator agreement to be reported.
- Recent uploads only; a single controversial video or live stream can move a channel's rate.
- Comments already removed by YouTube or the channel's moderators are invisible to the API, so these figures describe what remains visible.

## Acknowledgements

Started as a project at SRM Institute of Science and Technology under the guidance of Dr. Subalalitha C N.

## References

1. Obadimu, A., Mead, E., Hussain, M. N., & Agarwal, N. (2019). Identifying toxicity within YouTube video comments. *SBP-BRiMS 2019*.
2. Guberman, J., Schmitz, C., & Hemphill, L. (2016). Quantifying toxicity and verbal violence on Twitter. *CSCW '16 Companion*.
3. Hanu, L., & Unitary team (2020). Detoxify. GitHub, https://github.com/unitaryai/detoxify.
4. Rogan, W. J., & Gladen, B. (1978). Estimating prevalence from the results of a screening test. *American Journal of Epidemiology*, 107(1), 71–76.

## License

MIT © Harsh Raj
