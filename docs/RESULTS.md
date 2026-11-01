# Results, execution status and limitations

## Paper item mapping

|Item|Packaged inputs / model / records|Command and output|Status|
|---|---|---|---|
|Fig.1 and Table I|Raw graphs and selected Greedy/feature definitions; saved Fig.1 exports under results/paper/figures/fig1/|build-dataset / greedy create new graphs/labels; figures --figure 1 copies retained PNG previews|One intact graph and one Greedy case tested. Native editable diagram regeneration blocked by a private template; copying is not regeneration.|
|Fig.2|results/paper/held_out/IEEE300/history.csv; selected corrected checkpoint epoch150|figures --figure 2 writes PNG from 150 saved epochs|Replot executed successfully; new rendering/export code, not a byte-identical historical authoring claim.|
|Fig.3|Saved trafo_1_16 illustration, failed physical transformer1 at buses3-8; saved placement/exports; historical May checkpoint unavailable|figures --figure 3 copies existing exports|Export copy tested. Historical illustration inference/native re-render blocked; corrected models are not substituted.|
|Table II(a)|data/IEEE*/index.csv; data/splits/IEEE*/; models/held_out/IEEE*/; saved history/selection/test_results|verify / tables inspect and export checked coverage/K/counts; evaluate is an explicit new inference run|Saved139 test placements checked. One IEEE14 prediction agreed exactly in the newer environment. Full139case inference not rerun.|
|Table II(b)|Corrected source57 and target118 weights; fixed37target tests; results/paper/transfer/records.json|verify / tables inspect existing comparison; transfer runs a new frozen-source evaluation|All111 method records checked; full transfer inference unperformed.|
|Table II(c)|results/paper/runtime/raw-paired.csv and summary.json; January checkpoints held|tables inspects means; benchmark --budget stored/computed requires an explicit checkpoint|25092 saved rows inspected. Exact historical model rerun blocked; a corrected checkpoint would be a separately labeled new experiment.|
|Table II(d)|results/paper/ablation/raw-ablation.csv and summary.json; same held January models|tables exports A/B improvements; ablation requires explicit checkpoint|6273 saved rows retained;1394 A/Bplacements checked. No new ablation run; exact historical model rerun blocked.|

The [manifest](../results/paper/manifest.json) gives relative dataset, split, checkpoint, record paths and hashes for each supported population. Historical final-table authorship remains unconfirmed despite numerical consistency. Raw records and immutable retained PNG previews remain unchanged; four split metadata files have path-only portable representations with original scientific contents retained.

## Manuscript-reported values and saved-record calculations

The manuscript reports **95.12%** full coverage for IEEE300. The earlier audited
archive's 82 saved test placements instead yielded **9/82 = 10.9756097561%**
full coverage and **24515/24600 = 99.6544715447%** mean bus coverage. These
metrics differ, and the full-coverage disagreement remains unresolved.

In the current public checkout,
`results/paper/held_out/IEEE300/test_results.json` is absent although the
manifest requires it. Also, `results/paper/tables/held_out_saved_summary.csv`
differs from its manifest hash and now lists 78 fully covered IEEE300 cases.
That summary does not supply the missing per-scenario placement evidence.
Consequently, verification and verified table export cannot complete.
Resolve the source evidence and provenance explicitly; do not alter expected
counts, hashes, or raw records merely to match the manuscript.

## Executed checks

The checks below describe the earlier release-validation snapshot, not a claim
that the current public checkout passes artifact verification. Current missing
and changed evidence is identified above.

The base editable install succeeded in a new dedicated virtual environment using Python 3.12.14 and the bundled build tools (system-site packages available, existing environments unchanged). Import, CLI help for all commands, saved examples, the genuine Python verifier, table export, Fig. 1/3 export copying, experiment/output guards, alternate working directories and an independent copy in a path with spaces passed.

The base test run passed six tests and skipped four optional dependency tests. In an existing scientific environment, all ten tests passed. One real intact graph, one Greedy case, one IEEE14 grouped split and one prediction were checked. The generated intact x, edges, attributes, labels, bus IDs and graph targets matched the saved intact sample; split partitions, input hashes and class weights matched; the single prediction matched saved placement, probabilities and coverage. Model namespace isolation and callable training/evaluation/sparse factories were checked without training or benchmarks.

The isolated-copy verifier read its765 research input paths inside that copy, with original staging, original release, working project and private archive research paths blocked. The copied package therefore does not need its original drive location for saved inspection.

## Environment distinction and untested work

Recorded experiments used Python 3.11.4, NumPy 2.3.4, NetworkX 3.5, Torch 2.9.0+cpu, PyG 2.7.0, and runtime additionally pandapower 3.2.1. The scientific checks used Python 3.12.14, NumPy 2.4.6, NetworkX 3.6.1, Torch 2.9.1+cpu, PyG 2.8.0.post1, pandapower 3.5.4 and Matplotlib 3.11.1. These newer installed packages were not substituted into the historical environment record. Exact pinned-environment installation and full numerical reproduction remain untested.

No full training, full dataset generation, evaluation campaign, transfer campaign, timing suite or ablation was launched. No solver was run. Additional historical C/D/E ablation controls remain saved-only; the public new-experiment driver exposes A/B. Runtime driver scheduling/overhead differs from the original measurements. Native figure dependencies and some model binaries remain outside the release. No download URLs are invented.

Rights review remains unresolved for some historical/third-party artifacts; no first-phase license is automatically imposed. Source references were converted to portable modules, not represented as files literally used in the original experiment. Exact original generator environment, final table authorship, publication completion and the saved-record comparison remain documented.

Optional EMF/PDF figure exports are intentionally archived outside this release, not missing required inputs. The manifest retains their hashes and archive status, while all four required PNG previews remain subject to normal hash verification. Figure generation inputs and numerical records are preserved.
