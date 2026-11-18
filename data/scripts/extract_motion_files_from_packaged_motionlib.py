# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Extract per-motion .motion files from a packaged MotionLib .pt file."""

from pathlib import Path
from typing import Dict

import torch
import typer
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_enable=False)


def _resolve_flat_output_filename(
    original_motion_file: str,
    seen_names: Dict[str, int],
) -> str:
    """Return a collision-safe flat filename (no subdirectories)."""
    base_name = Path(original_motion_file).name.replace(" ", "_")
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix or ".motion"

    count = seen_names.get(base_name, 0)
    if count == 0:
        seen_names[base_name] = 1
        return f"{stem}{suffix}"

    seen_names[base_name] = count + 1
    return f"{stem}__dup_{count}{suffix}"


def _validate_required_fields(data: Dict[str, torch.Tensor]) -> None:
    required_fields = [
        "gts",
        "grs",
        "gvs",
        "gavs",
        "dvs",
        "dps",
        "length_starts",
        "motion_num_frames",
        "motion_dt",
        "motion_files",
    ]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field in packaged file: '{field}'")


@app.command()
def main(
    input_motion_lib_file: Path = typer.Argument(
        ..., help="Path to the packaged MotionLib .pt file."
    ),
    output_dir: Path = typer.Option(
        ..., help="Directory where extracted .motion files will be written."
    ),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Overwrite existing .motion files."
    ),
):
    """Extract individual .motion files from a packaged MotionLib file."""
    if not input_motion_lib_file.is_file() or input_motion_lib_file.suffix != ".pt":
        raise typer.BadParameter(
            f"Input must be a .pt file, got: {input_motion_lib_file}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading packaged motion lib: {input_motion_lib_file}")
    packaged = torch.load(input_motion_lib_file, map_location="cpu", weights_only=False)
    _validate_required_fields(packaged)

    motion_files = tuple(str(p) for p in packaged["motion_files"])
    motion_count = len(motion_files)
    if motion_count == 0:
        print("No motions found in packaged file.")
        return

    if packaged["length_starts"].numel() != motion_count:
        raise ValueError(
            "length_starts size does not match number of motion_files: "
            f"{packaged['length_starts'].numel()} vs {motion_count}"
        )
    if packaged["motion_num_frames"].numel() != motion_count:
        raise ValueError(
            "motion_num_frames size does not match number of motion_files: "
            f"{packaged['motion_num_frames'].numel()} vs {motion_count}"
        )
    if packaged["motion_dt"].numel() != motion_count:
        raise ValueError(
            "motion_dt size does not match number of motion_files: "
            f"{packaged['motion_dt'].numel()} vs {motion_count}"
        )

    total_frames = packaged["gts"].shape[0]

    written = 0
    skipped = 0
    seen_names: Dict[str, int] = {}

    for i in tqdm(range(motion_count), desc="Extracting motions"):
        start = int(packaged["length_starts"][i].item())
        num_frames = int(packaged["motion_num_frames"][i].item())
        end = start + num_frames

        if start < 0 or end > total_frames or num_frames <= 0:
            raise ValueError(
                f"Invalid frame range for motion {i}: start={start}, "
                f"num_frames={num_frames}, end={end}, total_frames={total_frames}"
            )

        dt = float(packaged["motion_dt"][i].item())
        if dt <= 0:
            raise ValueError(f"Invalid motion_dt for motion {i}: {dt}")

        output_filename = _resolve_flat_output_filename(motion_files[i], seen_names)
        out_path = output_dir / Path(output_filename).with_suffix(".motion")

        if out_path.exists() and not force_overwrite:
            skipped += 1
            continue

        motion_dict = {
            "fps": 1.0 / dt,
            "rigid_body_pos": packaged["gts"][start:end].clone(),
            "rigid_body_rot": packaged["grs"][start:end].clone(),
            "rigid_body_vel": packaged["gvs"][start:end].clone(),
            "rigid_body_ang_vel": packaged["gavs"][start:end].clone(),
            "dof_pos": packaged["dps"][start:end].clone(),
            "dof_vel": packaged["dvs"][start:end].clone(),
        }

        if "contacts" in packaged and packaged["contacts"] is not None:
            motion_dict["rigid_body_contacts"] = packaged["contacts"][start:end].clone()

        if "lrs" in packaged and packaged["lrs"] is not None:
            motion_dict["local_rigid_body_rot"] = packaged["lrs"][start:end].clone()

        torch.save(motion_dict, out_path)
        written += 1

    print(f"Extracted motions written: {written}")
    print(f"Skipped existing files: {skipped}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    app()