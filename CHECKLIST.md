# Push-T project checklist

Workflow: finish a phase, then stop. Phases: 1 data · 2 training · 3 simulation demo · 4 real robot. The next phase starts only when you say **"proceed"**.
Legend: `[x]` done · `[ ]` to do · 👤 = needs you (the human)

## Phase 1: Data collection pipeline ✅ built and tested; now collecting

### Environment
- [ ] uv + Python 3.12 venv in `.venv` (no Docker, no sudo)
- [ ] LeRobot 0.6.1 (dataset, training, diffusion), MuJoCo 3.14, torch 2.11 + CUDA (RTX 4050)
- [ ] Versions pinned in `requirements.lock`
- [ ] GPU offscreen rendering via EGL on NVIDIA (0.8 ms/frame)
- [ ] 👤 optional: `sudo apt install ffmpeg` (enables the faster torchcodec decoder for phase 2)

### Simulation
- [ ] Scene: Panda + stick, table, T block (free body), fixed gray target (`pusht/scene.py`)
- [ ] Planar IK controller: z locked (±1 mm), stick vertical (<0.3°), workspace clamp, 0.25 m/s limit
- [ ] Gravity compensation + tuned servo damping (tip lag about 8 mm at full speed)
- [ ] Env: seeded random starts, coverage metric, success = 90 % coverage for 0.5 s (≈5 mm / 5°; configurable) (`pusht/env.py`)
- [ ] Cameras from YAML: `overhead`, `corner`, `corner_left`, `top`, `side`, `wrist`

### Recording
- [ ] Raw HDF5 per episode: full sim state @50 Hz + actions + configs + physics XML
- [ ] Atomic writes (crash-safe), resumable sessions, discard flag
- [ ] Mouse teleop window at 50 Hz with HUD, coverage bar, auto-save on success
- [ ] Review tool: summary table, replay with any camera, mark discarded
- [ ] Exporter → LeRobot v3 (any cameras / resolution / fps, optional lighting randomization)
- [ ] Validator: structure, stats, decoded frames, action/state alignment, open-loop replay

### Verification (`scripts/smoke_test.py`: 23/23 pass)
- [ ] Controller sweep: all workspace corners reached, z drift 0.95 mm, orientation 0.25°
- [ ] Scripted demos recorded through the same recorder, all succeed
- [ ] Re-render from stored state reproduces the block pose exactly (error 0)
- [ ] Re-simulating recorded setpoints reproduces the full trajectory exactly (error 0)
- [ ] Same raw data exported with 2 different camera setups (96×96 @10 fps and 120×160 @25 fps)
- [ ] Both datasets load in `LeRobotDataset` and pass validation
- [ ] Teleop window opens on the real display at about 49 Hz

### 👤 Your part before phase 2
- [ ] 👤 Set the real T dimensions and target placement in `configs/scene.yaml` (**before** collecting)
- [ ] 👤 Re-run `scripts/smoke_test.py` after any change to `scene.yaml`
- [ ] 👤 Trial run: 3–5 demos with `scripts/02_collect.py` to get a feel for it, then tell me about any issues
- [ ] Camera study on 86 demos (`scripts/07_camera_study.py`): fixed cam + `wrist` never both blind
- [ ] 👤 Decide cameras for training (depends on whether the real Franka gets a wrist camera)
- [ ] 👤 Collect about 200 successful demos (86 so far) (`scripts/03_review.py --summary` shows the count)
- [ ] 👤 Review and discard bad demos (`scripts/03_review.py`, key `D`)
- [ ] 👤 Back up `data/raw/`
- [ ] Export `data/lerobot/pusht_sim` + validate (I can do this when you say proceed)
- [ ] Say **"proceed"** → phase 2

## Phase 2: Diffusion Policy 
- [ ] Cameras chosen: `overhead` + `wrist` (camera study), dataset exported + validated
- [ ] Low-dim dataset (`--no-images`) for the sanity-check policy
- [ ] `training/` built: configs, `train.py` (LeRobot, GPU, AMP, resume), `evaluate.py`, `policy_runner.py`
- [ ] Pipeline test: 400-step run → checkpoint → closed-loop sim evaluation + videos work
- [ ] GPU sizing: paper U-Net (278M) runs out of memory on 6 GB → `[256, 512, 1024]` (89M, 2.3 GB)
- [ ] Sanity baseline `dp_lowdim` (30k steps, ~1 h) → evaluate  (first attempt stopped at 5k steps, rerun with `--overwrite`)
- [ ] 👤 Main policy `dp_image` trained (100k steps, final loss 0.004)
- [ ] 👤 `evaluate.py outputs/dp_image --all` (13 min): success 76–84 % from 40k steps on (plateau).
      Best: 080000 (84 %), 050000 (82 %). Most failures stall at 50–80 % coverage, 1 near miss.
      ≥95 % = 0 because episodes stop at the 90 % rule (demos also ended at ~90 %)
- [ ] DDIM-10 sampler on 080000: 80 % (vs 84 % DDPM, within noise), 8× faster → default for real time
- [ ] 100-episode evaluation of 080000: DDPM 74 % [95 % CI 65–82], DDIM-10 78 % [69–85]; together 152/200 = 76 %.
      Same starts run twice disagree on 22 % (diffusion sampling noise). Eval starts are as far from the demos
      as demos are from each other (median 3.0 vs 3.2 cm); success does not drop with distance (`training/analyze_eval.py`)
- [ ] 👤 Collect more demos (86 → ~200), re-export, retrain (60k steps is enough), re-evaluate
- [ ] Iterate if success is low (more demos, lighting randomization, longer training)
- [ ] Stop and wait for **"proceed"**

## Phase 3: Simulation demo (inference showcase) 
Goal: watch the best trained policy work in the simulator through a clear UI, before any real hardware.
- [ ] Best model chosen automatically from evaluation results (`best_checkpoint`) → 080000, fast DDIM-10 sampler
- [ ] Demo app `demo/sim_demo.py`: live view with 8 camera tabs + 2×2 grid, free orbit camera, what-the-policy-sees panel
- [ ] Live info: plan overlay, coverage bar + graph over time, session success history, planning time
- [ ] Clickable buttons + keyboard shortcuts; drag / rotate / kick the T to test the policy
- [ ] Record the window to MP4; switch checkpoint and sampler live
- [ ] Tested (automated): ~50 fps (planning on a background thread), drag, grid, recording, checkpoint/sampler switch
- [ ] 👤 Try it and give feedback
- [ ] Stop and wait for **"proceed"**

## Phase 4: Real robot (not started)
- [ ] Real stick tool on the Franka, T block and target outline on the table
- [ ] Real cameras placed to match the chosen sim cameras (calibrate the table frame)
- [ ] xy → Franka Cartesian impedance controller with z locked, same limits and 10 Hz action rate
- [ ] Real teleop data collection in the same LeRobot format (fine-tuning / co-training)
- [ ] Deploy policy, evaluate, iterate
