#!/usr/bin/env python3
"""
Recover oss_annotations.json, oss_coord_adjustments.json, and review_status.json
from OSS overlays. Use this if local files were lost/corrupted.

Usage:
    export OSS_ENDPOINT=... OSS_ACCESS_KEY_ID=... OSS_ACCESS_KEY_SECRET=... OSS_BUCKET_NAME=...
    python recover_from_oss.py [--folders recordings_0303,recordings_0318]
"""

import json
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Recover annotations from OSS overlays')
    parser.add_argument('--folders', default='recordings_0303,recordings_0318',
                        help='Comma-separated list of OSS folders to scan')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be recovered without writing')
    args = parser.parse_args()

    try:
        import oss_client
        import oss2
    except ImportError:
        print("Error: oss_client.py and oss2 package required")
        sys.exit(1)

    bucket = oss_client._get_bucket()
    folders = [f.strip() for f in args.folders.split(',')]

    oss_annotations = {}
    coord_adjustments = {}
    review_status = {}

    total_overlays = 0
    total_with_mark = 0

    for folder in folders:
        ann_folder = folder + '_annotations'
        print(f"\nScanning {ann_folder}/...")

        for obj in oss2.ObjectIteratorV2(bucket, prefix=ann_folder + '/'):
            if obj.is_prefix():
                continue
            if not obj.key.endswith('/overlay.json'):
                continue

            # Extract recording name from path: {folder}_annotations/{rec_name}/overlay.json
            parts = obj.key.split('/')
            if len(parts) < 3:
                continue
            rec_name = '/'.join(parts[1:-1])  # everything between folder_annotations/ and /overlay.json
            ann_key = f"{folder}/{rec_name}"

            try:
                data = json.loads(bucket.get_object(obj.key).read().decode())
            except Exception as e:
                print(f"  Error reading {obj.key}: {e}")
                continue

            total_overlays += 1

            # Extract coord_adjustments
            case_coords = data.pop('coord_adjustments', {})
            for step_idx, adj in case_coords.items():
                coord_adjustments[f"{ann_key}_{step_idx}"] = adj

            # Store annotation
            oss_annotations[ann_key] = data

            # Build review status from mark
            mark = data.get('mark')
            if mark == 'pass':
                review_status[ann_key] = 'reviewed'
                total_with_mark += 1
            elif mark == 'fail':
                review_status[ann_key] = 'rejected'
                total_with_mark += 1

            edits = len(data.get('justification_edits', {}))
            errors = len(data.get('step_errors', {}))
            status = data.get('annotator_status', '-')
            print(f"  {rec_name[:60]}... mark={mark or '-'} edits={edits} errors={errors} status={status}")

    print(f"\n{'='*60}")
    print(f"Recovery summary:")
    print(f"  Total overlays found: {total_overlays}")
    print(f"  With review mark: {total_with_mark}")
    print(f"  Coord adjustments: {len(coord_adjustments)}")
    print(f"  Review statuses: {len(review_status)}")

    if args.dry_run:
        print("\n[DRY RUN] No files written. Remove --dry-run to write files.")
        return

    # Write files
    print(f"\nWriting files...")

    with open('oss_annotations.json', 'w') as f:
        json.dump(oss_annotations, f, indent=2, ensure_ascii=False)
    print(f"  oss_annotations.json: {len(oss_annotations)} entries")

    with open('oss_coord_adjustments.json', 'w') as f:
        json.dump(coord_adjustments, f, indent=2, ensure_ascii=False)
    print(f"  oss_coord_adjustments.json: {len(coord_adjustments)} entries")

    with open('review_status.json', 'w') as f:
        json.dump(review_status, f, indent=2)
    print(f"  review_status.json: {len(review_status)} entries")

    print("\nRecovery complete!")


if __name__ == '__main__':
    main()
