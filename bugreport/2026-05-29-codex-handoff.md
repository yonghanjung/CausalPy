# CausalPy 작업 핸드오프 (→ Codex)

- 작성일: 2026-05-29 (CDT)
- 작성 주체: `causalpy` agent (Claude)
- 대상 repo: `/Users/yonghanjung/paios/research/project/CausalPy/` (자체 `.git`을 가진 standalone repo. parent PAIOS repo와 별개)
- 현재 작업 branch: `bugfix/minimal-fixes-2026-05-29` (base = `7056413e "Add CausalPy agent contract"`, HEAD = `f29f6318`, 총 14 commit, **전부 local · push 안 함**)
- 재현 환경: `~/.venvs/causalpy-py311` (Python 3.11.14, repo 밖)
- 이 핸드오프 문서 자체: `bugfix/minimal-fixes-2026-05-29` branch에 doc commit으로 올라가 있다. (작성 직후엔 untracked였으나, branch만 보고 작업하는 Claude Code가 볼 수 있도록 commit함.)

이 문서는 세 부분이다. **(1) Diagnosis**, **(2) 지금까지 한 일**, **(3) 앞으로 할 일**. Codex가 처음 보는 상태에서 바로 이어받을 수 있도록 환경·규칙·검증법까지 모두 적는다.

---

## 0. 가장 먼저 읽어야 할 경고 (CRITICAL)

이걸 모르면 멀쩡한 코드를 망가뜨린다.

1. **H06 "당연한 수정"은 HARMFUL이다. 적용 금지.** `est_mSBD.py`의 sequential IPW가 stage 가중치를 곱으로 누적하지 않고 마지막 stage만 쓰는 게 "버그처럼" 보이고, 코드에 `# Note: This line preserves the bug` 주석까지 있다. 하지만 실제로 `=`→`*=`(누적곱)으로 바꾸면 IPW가 $P=1.5, 1.78$ 같은 **확률>1** 값을 내고 MAE가 $0.06 \to 0.43$으로 악화된다. 이유: `statmodules.sequential_quadratic_balancing`이 돌려주는 stage별 가중치는 각자 $\sum w = n$으로 정규화된 entropy-balancing weight라서 곱하면 정규화가 깨진다. `*=` 수정은 검증상 harmful이므로 금지. 현재 last-stage 사용은 기존 balancing normalization과 맞고 valid 범위([0,1]) 답을 준다. 다만 이것이 통계적으로 옳은 estimand인지는 paper derivation으로 확정해야 하며, **수식 변경은 그 전까지 금지**. 정적 분석만으로 판단하지 말고 반드시 ground_truth와 수치 비교할 것.
2. **Estimator는 seed를 줘도 재현되지 않는다 (BUG-N02).** `learn_mu`/`learn_pi`의 XGBoost params에 `n_jobs/nthread=4`가 박혀 있어 병렬 float reduction 때문에 run마다 추정값이 ~0.003 흔들린다. `OMP_NUM_THREADS=1`만으로는 안 잡힌다(params가 thread 수를 직접 지정). 따라서 **fix 전/후 bit-identical 비교는 불가능**하다. 검증은 (a) ground_truth anchor + 여러 run 평균, 또는 (b) noise-free isolation(아래 §3.D) 으로 한다.
3. **OSQP가 간헐적으로 `"solved inaccurate"`를 낸다.** N03에서 이걸 fatal 처리하던 걸 고쳤지만, 데이터 conditioning에 따라 estimation이 여전히 다른 이유로 멈출 수 있다. n을 키우면(예: 3000) 더 자주 본다.
4. **clustered 그래프는 `cluster_map`이 필수다.** `BD_SCM(d=4)` 같은 예제는 그래프에 conceptual node `C`가 있고 데이터에는 `C1..C4` 컬럼이 있다. estimator를 `cluster_map` 없이 부르면 `KeyError: ['C'] not in index`. 항상 `cm = graph.build_cluster_map(graph.find_topological_order(G), obs)`를 만들어 넘긴다.
5. **노드 이름 접두사 규약에 코드 전체가 의존한다.** treatment `X*`, outcome `Y*`, latent `U*`, 기타 `V*`. 이 규약을 벗어난 이름을 쓰면 `find_topological_order`(U 제외), c-component 탐지(U로 latent 판별), `tian.py`의 `x.startswith('X')` 등이 조용히 틀린다.
6. **WIP을 건드리지 마라.** repo는 이미 dirty하다(아래 §1.4). `examples.py`, `random_generator.py`, `test_ID_proportion.py`, `tian.py`, `tmp.py`는 사용자의 미커밋 변경이라 **절대 수정·커밋하지 않는다**. 이 파일들을 고쳐야 하는 버그(H08/M18/M16)는 사용자가 WIP을 정리한 뒤 별도 라운드에서 한다.

---

## 1. Diagnosis (코드 상태 진단)

### 1.1 CausalPy가 무엇인가

ADMG(latent를 `U_*` 노드로 표현하는 acyclic directed mixed graph)에서 인과효과 $P(Y \mid do(X))$를 다루는 연구용 라이브러리. 설계상 **Identification engine과 Estimation engine을 분리**한다.

- **graph layer** (`graph.py`): ADMG를 `nx.DiGraph` 하나로 표현. bidirected는 `U_A_B -> A`, `U_A_B -> B` canonical 인코딩. c-component, ancestor/descendant, edge cut($G_{\bar X}$, $G_{\underline X}$), random ADMG 생성.
- **identification layer**: `identify.py`(ID 알고리즘, c-tree/AC-tree, hedge 비식별), `adjustment.py`(backdoor/adjustment criterion), `mSBD.py`(sequential backdoor SAC), `frontdoor.py`(constructive FD), `tian.py`(Tian/generalized Tian, MC/WMC). 식별 가능성을 판정하고 symbolic estimand 문자열/LaTeX를 만든다.
- **estimation layer**: `est_mSBD.py`(**현재 유일하게 살아있는 추정기**: BD/SBD/mSBD에 대한 OM/IPW/DML cross-fitting), `est_general.py`/`est_plugin.py`/`est_Tian.py`(AC-tree c-component/Tian 추정기, **현재 import/호출 시 죽음**), `statmodules.py`(XGBoost nuisance, OSQP balancing, ground truth, 성능), `UCA.py`.
- **SCM/data layer**: `SCM.py`(구조방정식·샘플링), `example_SCM.py`(named ground-truth SCM 약 26종), `examples.py`(graph fixture), `random_generator.py`(random ADMG/SCM 탐색).
- **experiment layer**: `simulation.py` + `load_*.py` (Monte-Carlo 추정기 비교·플롯).

### 1.2 한 문장 진단

**Identification layer는 대체로 일관되게 살아있다. Estimation/simulation layer는 `est_mSBD.py` 리팩터가 상위 모듈로 전파되지 않아 다수가 실행 불가다.** 그 외에 작은 버그가 여럿 있고, 검증 과정에서 새 버그(N01~N03)를 발견했다.

### 1.3 버그 카탈로그 (정본: `bugreport/2026-05-29-bugs.md`)

상세 카탈로그는 `bugreport/2026-05-29-bugs.md`에 있다. 그 문서의 §0.1 "검증 후 정정"이 **가장 최신·정확한 판정**이다. 핵심만 요약:

- **A군 (BUG-H01~H04) = 가장 큰 단일 이슈.** `est_general.py`/`est_plugin.py`/`est_Tian.py`가 존재하지 않는 `est_mSBD.estimate_mSBD_xval_yval`, 없는 모듈 `est_BD`, 없는 함수 `graph.unfold_graph_from_data`를 호출한다. 이게 풀려야 c-component/Tian 추정과 simulation dispatcher가 돈다. (상세는 §3.A)
- **H05, H06: 정적 판정이 틀렸던 항목.** H05(`partition_Y` 인덱스)는 출력 무변경(benign), H06(IPW 누적)은 버그 아님 + 제안 수정이 harmful. 둘 다 환경 검증으로 정정됨.
- **새 발견 N01~N03:** N01(DML ≡ IPW, AIPW 보정 무작동), N02(estimator 비재현성), N03(OSQP `solved inaccurate` fatal — **이미 수정함**).
- 그 외 Medium/Low 다수. 이번 라운드에 처리한 것과 남은 것은 §2, §3 참조.

### 1.4 repo / 환경 사실관계 (중요)

- **CausalPy는 자체 `.git`을 가진 standalone repo.** parent PAIOS repo와 commit/push는 별도 승인.
- **repo가 이미 dirty하다 (사용자 WIP, 내가 만든 것 아님):** `examples.py`(+61), `random_generator.py`(+36), `test_ID_proportion.py`(+6), `tian.py`(+6), `tmp.py`(+475 대규모), 두 개의 "conflicted copy" 파일 삭제, 그리고 `.DS_Store` 수정.
- **절대 stage/commit 금지 목록:** 위 5개 WIP 소스, conflicted-copy 삭제, `.DS_Store`, tracked legacy venv 폴더 `CausalPy/CausalPy/`, 그리고 handoff/doc 파일(명시적으로 doc commit할 때만 예외). commit은 항상 `git add <고친 단일 소스 파일>`로 path-scoped. `git add -A` / `git add .` 절대 금지.
- **시스템 Python은 3.14**이고 deps가 하나도 없다. pinned deps(numpy 1.26.2 등)는 3.14 wheel이 없어 설치 불가. 그래서 **3.11이 필요**하다(`__pycache__`가 cpython-311/312였음 = 원래 3.11/3.12에서 돌던 코드).
- **`requirements.txt`에 `pyvis`, `dill`, `tabulate`, `osqp`가 빠져 있었다.** `graph.py`/`SCM.py`가 top-level에서 `from pyvis.network import Network`를 하므로, 이 4개가 없으면 `import graph`조차 실패한다 = 라이브러리 전체 import 불가. (M17에서 pin 추가함)
- **repo 안의 `CausalPy/CausalPy/` 폴더는 tracked legacy venv다.** `CausalPy/bin/python`은 존재하고 Python 3.11.14로 실제 실행된다(내 첫 진단의 "python 없음/깨진 venv"는 경로 오타로 인한 오판이었다). 다만 (a) git에 31,611개 파일이 tracked되어 있고 (b) `pyvenv.cfg`의 `command`가 Dropbox 원본 경로(`/Users/yonghanjung/Dropbox/Personal/Research/Code/CausalPy/CausalPy`)를 가리킨다. **이 venv는 쓰지 말고**, repo 밖 `~/.venvs/causalpy-py311`를 유일한 authority로 삼는다. 이 폴더 안 파일은 절대 stage하지 않는다(이미 tracked라 실수로 변경이 commit될 위험).

### 1.5 가장 중요한 교훈

**정적 분석은 estimator 동작을 자주 틀리게 판단한다.** H06가 산 증인이다("당연한 수정"이 estimator를 망가뜨림). estimator 수식·식별 로직을 바꿀 때는 반드시 실행 환경에서 ground_truth와 수치 비교할 것. 작은 typo/config/crash 수정은 정적으로도 안전하지만, 수치를 바꾸는 수정은 환경 검증이 필수다.

---

## 2. 지금까지 한 일

### 2.1 재현 환경 구축 (재생성 방법 포함)

repo 밖에 Python 3.11 venv를 만들고 pinned deps + 누락 4종을 설치했다.

```bash
# uv가 /opt/homebrew/bin/uv 에 있고 python3.11이 /opt/homebrew/bin/python3.11 에 있음
uv venv --python 3.11 ~/.venvs/causalpy-py311
cd /Users/yonghanjung/paios/research/project/CausalPy
uv pip install --python ~/.venvs/causalpy-py311/bin/python -r requirements.txt
# (requirements.txt에는 M17으로 pyvis/dill/tabulate/osqp가 이미 추가돼 있음)
```

설치 확인된 버전: numpy 1.26.2, scipy 1.11.4, pandas 2.1.4, sklearn 1.4.0, networkx 3.2.1, xgboost 2.0.3, pyvis 0.3.2, dill 0.4.1, osqp 1.1.1, tabulate 0.10.0.

**코드 실행법** (반드시 repo 디렉터리에서, matplotlib는 Agg 백엔드로):

```bash
cd /Users/yonghanjung/paios/research/project/CausalPy
MPLBACKEND=Agg ~/.venvs/causalpy-py311/bin/python - <<'PY'
import warnings, numpy as np, random; warnings.filterwarnings("ignore")
import example_SCM, graph, statmodules, est_mSBD
scm, X, Y = example_SCM.BD_SCM(seednum=42, d=4); G = scm.graph
obs = scm.generate_observational_samples(3000)
cm = graph.build_cluster_map(graph.find_topological_order(G), obs)
truth = statmodules.ground_truth(scm, X, Y, np.ones(len(Y), int))
ATE,VAR,lo,hi = est_mSBD.estimate_BD(G, X, Y, obs, cluster_map=cm, seednum=42)
print("truth", truth); print("OM", ATE['OM'])
PY
```

스크립트를 파일로 만들어 돌릴 거면 `PYTHONPATH=.`를 줘야 한다(파이썬이 cwd가 아니라 스크립트 디렉터리를 path에 넣기 때문). heredoc(stdin)으로 주면 cwd가 path에 들어가 그대로 import된다.

### 2.2 작업 방법론 (이 branch의 계약)

- **one bug, one commit.** 각 commit은 재현·수정·검증 근거를 가진다.
- **path-scoped commit.** `git add <단일 파일>` 후 commit. 사용자 WIP 5개 파일은 절대 staging/commit하지 않는다.
- **verify-before-commit.** 수치를 바꾸는 수정은 환경에서 before/after 검증. typo/config/crash는 syntax + 재현으로.
- **estimator 수식·identification 로직·API redesign은 이 branch에 섞지 않는다.**
- **push 안 함.** 전부 local commit.

### 2.3 commit 14개 (검증 근거 포함)

base `7056413e` 이후, 시간순:

| # | hash | bug | 변경 | 검증 |
|---|---|---|---|---|
| 1 | `5061f95e` | — | bug report 추가(doc) | — |
| 2 | `154c93c8` | M03 | `statmodules.py` `booser`→`booster` | syntax + 명백 |
| 3 | `49eb09a8` | H09 | `statmodules.py` `subsample 0.0`→`1.0` (learn_pi/learn_multi_pi) | syntax (값=1.0, learn_mu 기본값과 일치) |
| 4 | `37b07cf9` | M01 | `statmodules.py` OSQP 제약 0/0 가드(`_ratio`) | by-construction (target≠0이면 기존과 완전 동일, target==0만 nan→0) |
| 5 | `07014ca5` | H07 | `simulation.py:629` driver `'IW'`→`'IPW'` | 두 driver/compute_performance 키 일치 |
| 6 | `da51abd2` | M17 | `requirements.txt` +pyvis/dill/tabulate/osqp | 설치 resolve + `import graph` OK |
| 7 | `d48893a4` | H05 | `mSBD.py:33` `partition[f'Y{i}']`→`f'Y{j}']` | **benign 확인** (mSBD_SCM, mSBD_SCM_JCI, 3-treatment 합성에서 buggy==fixed) |
| 8 | `4427dda4` | M05 | `est_mSBD.py:412` `obs_data.copy()` + 라벨 `'Y'`→`'__indicator_outcome__'` | caller 'Y' 컬럼 추가 사라짐 + 추정값 noise 내 |
| 9 | `8c624e9a` | — | bug report 정정(doc): H05/H06 강등 + N01~N03 추가 | — |
| 10 | `38db7188` | N03 | `statmodules.py:106` OSQP `'solved inaccurate'` 수용 | mock before-raise/after-return + n=3000 완주 |
| 11 | `3a07337b` | M04 | `statmodules.py` compute_performance를 key로 정렬 비교 | 뒤집힌 dict MAE 0.8→0.0, 정상 0.1 |
| 12 | `ba567f31` | M07 | `graph.py` find_successors 정렬 후 검사 + `raise ValueError` | 정상통과/ValueError (참고: 호출처 없음) |
| 13 | `4773adb9` | M14 | `load_random_simluation.py:82` CoW-safe rename | CoW 켜고 before(IPW유지)/after(IW) |
| 14 | `f29f6318` | M02 | `statmodules.py` 죽은 `extract_error_bars` 제거 | rg 호출처 0 + import OK |

코드 변경 총량은 7개 소스 파일 약 24줄(나머지는 doc). 모든 commit이 단일 파일.

### 2.4 review 결과

push 전 14-commit 전수 review를 했다. 판정: **PASS.** one-bug-one-commit·minimal·WIP무침입·H06무혼입 확인. reviewer sign-off 필요한 "경계 항목" 5개:
1. **H05**: identification-layer(mSBD)를 건드림. 단 출력 무변경 검증됨.
2. **M17**: deps pin이 별도 hygiene branch로 가야 하는지(원칙상 환경 정리는 분리).
3. **H09**: `subsample` 값 `1.0` 선택 confirm.
4. **M04**: 이제 `ATE`에 truth key 없으면 `KeyError`(이전엔 조용히 misalign) — edge-case 동작 변화.
5. **M01**: 유일하게 numeric run 없이 by-construction 검증(N02 때문에 bit-identical 비교 불가).

### 2.5 의도적으로 안 한 것

- **H06 수정 안 함** (harmful).
- **A군(H01~H04) 손 안 댐** (큰 작업, 별도 계획).
- **N01(DML), N02(재현성) 손 안 댐** (deep, paper/설계 필요).
- **dirty 파일(examples/random_generator/test_ID_proportion/tian/tmp) 손 안 댐** (사용자 WIP).
- **push/merge 안 함.**

---

## 3. 앞으로 할 일

우선순위 순. 각 항목에 "무엇을/왜/어떻게 검증"을 적는다. 모든 작업은 §2.2 방법론과 §0 경고를 따른다.

### 3.A A군 — estimation 상위 모듈 재배선 (가장 큰 작업, Codex 주력 후보)

**증상.** `est_general.py`, `est_plugin.py`, `est_Tian.py`가 import/호출 시 죽는다.

**구체 결함 (grep으로 확인됨):**
- `est_general.py:24` `import est_BD` → `est_BD.py` 없음 → `ModuleNotFoundError` (import 자체 실패). 게다가 본문에서 안 쓰임 → 그냥 삭제.
- `est_general.py:370,407,427,435,533,644,659`, `est_plugin.py:98,133,150,157,334,350,413,435,482`, `est_Tian.py:70,117,133,196,218` → `est_mSBD.estimate_mSBD_xval_yval(...)` 호출. **이 함수는 어디에도 없다.**
- `est_general.py:451,607,673` → `graph.unfold_graph_from_data(...)` 호출. **graph.py에 없다** (cluster 처리는 `graph.expand_variables`/`graph.build_cluster_map`로 바뀜).
- `est_general.py:845-855` `estimate_case_by_case` (simulation이 호출하는 dispatcher): kwarg `cluster_variables`(실제는 `cluster_map`) → `TypeError`; `est_mSBD.estimate_mSBD(...)` 호출(없음, `estimate_mSBD_y`/`estimate_mSBD_xy`만 존재) → `AttributeError`; `only_OM=False` 하드코딩.

**현재 살아있는 새 API (`est_mSBD.py`):**
```
estimate_BD(G, X, Y, obs_data, alpha_CI=0.05, cluster_map=None, n_folds=2, seednum=123, only_OM=False)
estimate_SBD(...)  # 같은 시그니처
estimate_mSBD_y(G, X, Y, y_policy, obs_data, ..., cluster_map=None, ...)
estimate_mSBD_xy(G, X, Y, x_policy, y_policy, obs_data, ..., cluster_map=None, ...)
```
반환은 모두 `(ATE, VAR, lower_CI, upper_CI)`이고 각각 `{'OM','IPW','DML'} -> 값` 형태의 dict다. (단, `estimate_mSBD_xy`는 x_val 하나라 값이 dict가 아니라 estimator→scalar.)

**옛 호출부가 기대하는 형태:** `Q_xxx, _, _, _ = est_mSBD.estimate_mSBD_xval_yval(G, PA, S, pa_val, s_val, obs_data, ...)` 즉 **per-cell 단일 scalar Q**를 받아 `Q_D *= Q_Di` 같은 scalar 산술을 한다.

**왜 mechanical 재배선이 아니라 design 작업인가:** 옛 API는 "(PA → S) 부분문제의 단일 Q 값"을 돌려줬는데, 새 API는 estimator별(OM/IPW/DML) dict를 돌려준다. 그러니 (a) 어느 estimator를 plug-in Q로 쓸지(아마 OM/g-formula), (b) 반환 unpacking과 scalar 산술을 어떻게 맞출지 정해야 한다. 더해서 이 파일들 안에는 import 실패에 가려진 **latent 버그(LT01~LT07, bug report §3.4 참조)**가 있어, 재배선하면 바로 표면화된다(예: `est_general.py:667` `estimate_gTian`가 `**kwargs` 없이 `kwargs` 참조 → `NameError`; `est_plugin.py:273` `PA_S0` 미정의).

**Base branch.** 14-commit review를 통과한 `bugfix/minimal-fixes-2026-05-29`를 accepted baseline으로 보고, A군은 그 위에서 새 branch `bugfix/estimation-dispatcher-rewire-2026-05-xx`를 따서 시작한다. minimal-fix branch에 직접 추가하지 않는다. (대안: minimal branch를 먼저 merge/push한 뒤 `main`에서 새 branch를 따도 된다. 핵심은 A군을 minimal branch와 분리하는 것.)

**기계적 wrapper 금지 (semantic mismatch 위험).** 옛 `estimate_mSBD_xval_yval`는 임의의 factor `Q[S | do(PA)]` (단일 scalar)를 기대하는데, 새 `estimate_mSBD_xy`는 특정 causal effect/policy estimator다. 의미가 다르므로 `estimate_mSBD_xy`를 그대로 감싸는 shim을 대충 만들면 틀린다. **먼저 옛 API가 실제로 어떤 estimand였는지(어떤 그래프에서 어떤 Q-factor를 어떤 conditioning으로) 코드와 paper로 확인**하고, dispatcher 복구와 factor estimation 복구를 분리한다.

**3단계로 나눠서 (한 번에 다 하지 말 것):**
1. **1단계 — import 복구.** `est_general.py`가 import만 되게 한다. `import est_BD` 삭제(없는 모듈). import을 막는 최소한만 건드린다. 검증: `import est_general` 성공(import smoke).
2. **2단계 — BD/SBD/mSBD dispatcher만 복구.** `estimate_case_by_case`에서 backdoor / sequential-backdoor / mSBD 경로만 새 API(`estimate_BD`/`estimate_SBD`/`estimate_mSBD_y`)로 연결한다. `cluster_variables`→`cluster_map`, `only_OM` forward, `graph.unfold_graph_from_data`→`build_cluster_map`+`expand_variables`. **Tian/general c-component 경로는 아직 손대지 않는다.** 검증: BD/SBD/mSBD가 필요한 작은 SCM에서 `estimate_case_by_case` 실행 + `ground_truth` 비교(OM이 truth 근처, 값이 [0,1] 범위).
3. **3단계 — Tian / general c-component factor estimation.** 옛 `estimate_mSBD_xval_yval`의 estimand 의미를 확정한 뒤 **별도 설계**로 진행한다. **shim을 대충 만들지 말 것.** LT01~LT07(`est_general.py:667` `kwargs` 미정의, `est_plugin.py:273` `PA_S0` 미정의 등)도 이 단계에서 함께. 검증: `tian.check_Tian_criterion`이 True인 그래프(예: `example_SCM.Napkin_FD_SCM`, `Fulcher_FD`)에서 ground_truth 대조.

**각 단계 검증 공통:** import smoke + 작은 SCM example + `ground_truth` 비교. N02 때문에 여러 run 평균 또는 noise-free isolation(§3.D) 사용.

**절대 섞지 말 것:** estimator 수식 변경, DML/N01, H06, Tian factorization은 이 minimal dispatcher 복구와 같은 commit/branch에 넣지 않는다. dispatcher 복구가 끝난 뒤 각각 별도로.

### 3.B N01 — DML ≡ IPW (AIPW 보정 무작동), 깊은 분석

**증상.** `estimate_SBD`/`estimate_mSBD_y`/`estimate_mSBD_xy`에서 DML 추정값이 매 run IPW와 **정확히 동일**하다. AIPW의 doubly-robust 보정이 전혀 효과가 없다는 뜻.

**관련 코드** (`est_mSBD.py`, 예: `estimate_mSBD_y`):
```python
pseudo_outcome_dml = check_mu_preds[(1, x_val)].copy()  # = OM
for i in range(1, m + 1):
    pseudo_outcome_dml += pi_acc_dict[i] * (check_mu_preds[(i + 1, x_val)] - mu_preds[i])
# 그리고 IPW:
pi_accumulated = pi_preds[(m, x_val)]   # (덮어쓰기로 마지막 stage)
estimator_outcomes['IPW'] = pi_accumulated * Yvec
```
`pi_acc_dict[i] = pi_preds[(i, x_val)]`는 단일 stage 가중치(누적곱 아님). DML이 IPW와 정확히 같아지는 이유를 대수적으로 추적해야 한다(보정항이 telescoping으로 상쇄되어 IPW만 남는 듯).

**왜 minimal fix가 아닌가:** 올바른 sequential AIPW/DML 추정량은 sequential backdoor(SBD/mSBD)의 efficient influence function에서 유도된다. IPW(last-stage 가중치)는 검증상 valid 범위 답을 주고 기존 normalization과 맞지만, 그것이 옳은 estimand인지와 DML 보정의 올바른 형태는 둘 다 paper 기반 derivation으로 확정해야 한다. **Codex의 theorem-proving/symbolic derivation이 가장 잘 맞는 작업.** 

**참고 문헌(repo 안):** `research/material/`의 causal inference references, `research/papers/sca/`(sequential covariate adjustment), `research/papers/multilinear_causal_estimand/`. SBD/sequential DML의 EIF를 확인하고, `sequential_quadratic_balancing`이 만드는 가중치의 정확한 의미(각 $\pi^i$가 무엇을 균형 맞추는지)를 코드와 대조할 것.

**검증:** 올바른 DML이라면 (a) 값이 IPW와 달라야 하고, (b) nuisance가 약간 틀려도 OM/IPW보다 truth에 강건해야 한다. doubly-robust 성질을 의도적으로 한쪽 nuisance를 망가뜨려 테스트.

### 3.C N02 — estimator 비재현성

**증상.** seed 고정에도 추정값이 run마다 ~0.003 흔들림. 원인: `statmodules.learn_mu`(`'n_jobs': 4`)와 `learn_pi`/`learn_multi_pi`(`'nthread': 4`)의 XGBoost 병렬성.

**수정 방향(설계 결정 필요):** params에 `'seed': <고정>` 추가 + `n_jobs/nthread=1`(또는 `tree_method` 결정적 설정). 단 **속도 손해**가 있다. 연구·디버깅용 재현성 vs 속도 trade-off라, 옵션(`deterministic=True`)으로 노출하는 게 좋다. 이건 estimator 동작에 영향 주는 변경이므로 별도 branch + 검증.

**검증:** 수정 후 같은 seed로 두 번 돌려 bit-identical 확인.

### 3.D 검증 도구 모음 (Codex가 재사용할 것)

**(1) ground_truth anchor:**
```python
truth = statmodules.ground_truth(scm, X, Y, yval)  # yval = np.ones(len(Y), int) 등
# 내부적으로 X를 randomize(0.5 binomial)하고 1,000,000 표본으로 E[Y|do(x)] 계산.
# 주의: generate_samples(seed=None)을 써서 비재현(BUG-M09). 앞에서 np.random.seed/random.seed 고정할 것.
```

**(2) noise-free isolation (가중치 capture):** estimator 내부 가중치를 한 run에서 잡아 두 변형을 같은 입력으로 비교(아래는 IPW 비교 예; H06 검증에 실제로 썼던 방법):
```python
cap = []
orig = statmodules.sequential_quadratic_balancing
def wrapped(*a, **k):
    r = orig(*a, **k); cap.append((tuple(k['x_vals']), {kk: np.array(vv) for kk,vv in r.items()})); return r
statmodules.sequential_quadratic_balancing = wrapped
try:
    est_mSBD.estimate_mSBD_y(G, X, Y, [1,1], obs.copy(), cluster_map=cm, seednum=42)
finally:
    statmodules.sequential_quadratic_balancing = orig
# cap에 (x_val, {pi_1, pi_2, ...}) 들어옴. 같은 가중치로 여러 estimator 변형을 noise 없이 비교.
```

**(3) 결정적 데이터:** `np.random.seed(S); random.seed(S)` 직후 `scm.generate_observational_samples(n)` 호출하면 그 호출은 결정적(M09 우회). 단 estimator 자체는 N02 때문에 여전히 비결정적.

### 3.E dirty 파일 버그 (사용자 WIP 정리 후)

`random_generator.py`, `test_ID_proportion.py`는 사용자 WIP과 같은 파일이라 **WIP을 commit/stash로 정리한 뒤** 진행:
- **H08** `random_generator.py:61,83` `random.randint(0, 1e7)` → `int(1e7)`. `1e7`은 float이라 Python 3.12+에서 `TypeError`(3.11에선 DeprecationWarning). `max_graphs` 기본값 `1e7`(line 42)도 정리.
- **M18** `random_generator.py:85-94` `sparcity_constant`가 `if` 안에서만 바인딩 → 명시 kwarg + 2회차 반복 시 `UnboundLocalError`. 루프 전에 초기화.
- **M19** `random_generator.py`의 self-reseed로 탐색이 저엔트로피 궤적에 빠짐(`generate_random_graph`가 내부에서 전역 RNG reseed). 지역 RNG로 바꾸는 설계 변경.
- **M16** `test_ID_proportion.py:78` `Ratio` 버킷이 `satisfied_FD == False`를 빠뜨려 MECE 분할이 겹침. conjunction에 추가.

### 3.F 남은 Medium/Low (clean 파일, 다음 minimal 라운드 가능)

- **M06** `est_mSBD.py` `VAR` 필드 의미 불일치(`estimate_BD`는 표본분산+CI에서 /n, 나머지는 평균의 분산+CI에서 /1 → estimator 간 $n$배 차이). 통일하려면 어느 convention이 정본인지 결정.
- **M08** `mSBD.py` `check_mSBD`(range(1,..))와 `check_SAC`(range(0,..))의 $Y_0$ 누적 범위 불일치. paper 정의 확인 필요.
- **M09** `SCM.py:208` `generate_samples(seed=None)` 기본 경로 비재현(ground_truth/observational에서 사용). seed 일관 전달.
- **M10** `SCM.py:113` `binary_equation` 이중 노이즈 + raw sigmoid(`expit` 미사용).
- **위생**: `naive_estimator.py`(빈 stub), `identify.py:243-319`(존재하지 않는 함수 참조하는 주석 처리 dead code), `tmp.py`(`__main__` 가드 없이 import-time에 무거운 MC + blocking `plt.show()`), `graph.py:230`/`frontdoor.py:92` dead 변수.

### 3.G 즉시 결정이 필요한 것 (review 경계 항목, §2.4)

push 전에 사용자/Codex가 정해야 함: M17을 hygiene branch로 분리할지, H05를 이 branch에 둘지, H09 값(1.0) 확정, M04의 KeyError 동작 수용, M01의 by-construction 검증 수용. (전부 "그대로 두고 push"도 가능.)

---

## 4. 작업 규칙 (이어받는 누구든 지킬 것)

1. **one bug, one commit.** 재현·수정·검증 근거 필수.
2. **절대 stage/commit 금지:** 사용자 WIP(examples/random_generator/test_ID_proportion/tian/tmp), conflicted-copy 삭제, `.DS_Store`, tracked legacy venv `CausalPy/CausalPy/`, handoff/doc(명시적 doc commit 제외). commit은 `git add <단일 소스>`로 path-scoped만(`git add -A` 금지). dirty WIP은 사용자가 정리한 뒤 별도 라운드.
3. **estimator 수식·identification 로직·API redesign은 이 minimal-fix branch(`bugfix/minimal-fixes-2026-05-29`)에 섞지 않는다.** A군·N01·N02·M06·M08은 **별도 branch**.
4. **"당연해 보이는 수정"도 before/after 수치 검증(ground_truth 대조) 없으면 적용 금지.** H06가 경고.
5. **dead code 제거는 `rg`로 호출처 0건 + import smoke 통과일 때만.**
6. **dependency 수정은 실제 import 실패를 재현한 것만 pin.** 환경 정리는 별도 hygiene branch.
7. **push 전 전체 commit 전수 review** (각 commit이 minimal·무관 변경 없음 확인).
8. **estimation 검증은 항상 `~/.venvs/causalpy-py311`에서, `cluster_map` 넘기고, N02 비결정성을 감안(평균 또는 noise-free isolation).**

---

## 5. 한 줄 상태

`bugfix/minimal-fixes-2026-05-29`에 검증된 minimal bugfix 12개 + 문서 2개(총 14 commit, local). H06는 의도적으로 미수정(harmful). 다음 큰 일은 A군 재배선과 N01(DML) — 둘 다 별도 branch + 환경 검증 + (N01은) paper 기반 derivation 필요.
