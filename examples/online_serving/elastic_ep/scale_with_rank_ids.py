#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example script demonstrating how to use the scale_elastic_ep API
with shrinked_dp_rank_ids parameter to specify which specific
data parallel ranks to remove during scale down operations.
"""

import argparse
import json
import sys

import requests


def scale_with_rank_ids(host, port, new_dp_size, shrinked_dp_rank_ids=None):
    """
    Scale the elastic EP deployment with optional specific rank IDs to remove.

    Args:
        host: API server host
        port: API server port
        new_dp_size: New data parallel size
        shrinked_dp_rank_ids: List of rank IDs to remove (for scale down only)
    """
    url = f"http://{host}:{port}/scale_elastic_ep"
    payload = {"new_data_parallel_size": new_dp_size}

    # Add shrinked_dp_rank_ids if provided
    if shrinked_dp_rank_ids is not None:
        payload["shrinked_dp_rank_ids"] = shrinked_dp_rank_ids

    headers = {"Content-Type": "application/json"}

    print(f"Sending scale request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("Scale request successful!")
            return True
        else:
            print("Scale request failed!")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test scale functionality with specific rank IDs"
    )
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8006, help="API server port")
    parser.add_argument(
        "--new-dp-size", type=int, required=True, help="New data parallel size"
    )
    parser.add_argument(
        "--shrinked-rank-ids",
        type=int,
        nargs="*",
        help="Specific rank IDs to remove during scale down (space-separated)",
    )

    args = parser.parse_args()

    success = scale_with_rank_ids(
        args.host, args.port, args.new_dp_size, args.shrinked_rank_ids
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
