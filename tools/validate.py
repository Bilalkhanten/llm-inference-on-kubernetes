#!/usr/bin/env python3
"""Offline schema validation of the manifests against Kubernetes OpenAPI schemas (no cluster needed).

Usage: python3 tools/validate.py [K8S_VERSION]     e.g. 1.34 (default)

Built-in API groups are validated strictly. CRDs (KEDA, Gateway API, GAIE) have no bundled
schema, so they are only parsed as YAML and reported as SKIP.
Exit code is 1 if any built-in resource FAILS validation.
"""
import glob
import os
import sys

import kubernetes_validate as kv
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTIN_API_VERSIONS = {
    "v1", "apps/v1", "rbac.authorization.k8s.io/v1",
    "scheduling.k8s.io/v1", "resource.k8s.io/v1",
}


def validate_all(version):
    """Return a list of (file, kind, status, message). status in OK / FAIL / SKIP."""
    results = []
    for path in sorted(glob.glob(os.path.join(ROOT, "manifests", "[0-9]*.yaml"))):
        name = os.path.basename(path)
        with open(path) as fh:
            for doc in yaml.safe_load_all(fh):
                if not doc:
                    continue
                kind, api = doc["kind"], doc["apiVersion"]
                if api not in BUILTIN_API_VERSIONS:
                    results.append((name, kind, "SKIP", "CRD, no bundled schema (parsed as YAML only)"))
                    continue
                try:
                    kv.validate(doc, version, strict=True)
                    results.append((name, kind, "OK", f"valid against Kubernetes {version} (strict)"))
                except kv.SchemaNotFoundError:
                    results.append((name, kind, "SKIP", f"no schema for {api} in {version}"))
                except kv.ValidationError as e:
                    results.append((name, kind, "FAIL", e.message[:90]))
    return results


def main(argv):
    version = argv[1] if len(argv) > 1 else "1.34"
    results = validate_all(version)
    for name, kind, status, msg in results:
        print(f"{name:<28}{kind:<22}{status:<6}{msg}")
    return 1 if any(r[2] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
