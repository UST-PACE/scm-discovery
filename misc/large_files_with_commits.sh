#!/usr/bin/env bash

# ****************************************************************************************************************************
# DISCLAIMER: This script is provided "AS IS" without warranty of any kind.
# Use at your own risk. No liability for data loss, repo changes, or downtime.
# ****************************************************************************************************************************

# Dependencies: git, awk, and a POSIX shell (bash/zsh). For private repos, SSH auth must be set up.

# What it bash script does:
#   Scans all commits across all branches for large blobs in a GitLab repo mirror,
#   and outputs a CSV with branch, path, size, commit hash, and author.

# Execute:
#   chmod +x large_files_with_commits.sh
#   ./large_files_with_commits.sh \
#     --url https://gitlab.com/group/project.git \
#     --threshold-mb 100 \
#     --mirror-root ~/.gl-ghec-mirror \
#     --out /tmp/large_files_with_commits.csv

url=""
threshold_mb=""
mirror_root=""
out=""

usage() {
  cat <<'EOF'
Usage:
  ./large_files_with_commits.sh --url <gitlab_repo_url> --threshold-mb <mb> --mirror-root <dir> --out <csv>

Options:
  --url            GitLab repo HTTPS/SSH URL (required)
  --threshold-mb   Size threshold in MB (required)
  --mirror-root    Mirror root directory (required)
  --out            Output CSV path (required)
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      url="$2"
      shift 2
      ;;
    --threshold-mb)
      threshold_mb="$2"
      shift 2
      ;;
    --mirror-root)
      mirror_root="$2"
      shift 2
      ;;
    --out)
      out="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$url" ]]; then
  echo "Error: --url is required."
  usage
  exit 1
fi
if [[ -z "$threshold_mb" ]]; then
  echo "Error: --threshold-mb is required."
  usage
  exit 1
fi
if [[ -z "$mirror_root" ]]; then
  echo "Error: --mirror-root is required."
  usage
  exit 1
fi
if [[ -z "$out" ]]; then
  echo "Error: --out is required."
  usage
  exit 1
fi

threshold_bytes=$((threshold_mb * 1024 * 1024))

host="${url#https://}"; host="${host%%/*}"
path="${url#https://*/}"; path="${path%.git}"
mirror="${mirror_root}/${host}/${path}.git"

if [[ ! -d "$mirror" ]]; then
  mkdir -p "$(dirname "$mirror")"
  git clone --mirror "$url" "$mirror"
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

git -C "$mirror" for-each-ref --format='%(refname:short)' refs/heads | while read -r br; do
  git -C "$mirror" rev-list --objects "$br" | \
  git -C "$mirror" cat-file --batch-check='%(objecttype) %(objectsize) %(objectname) %(rest)' | \
  awk -v b="$br" -v t="$threshold_bytes" '
    $1=="blob" && $2>=t {
      path = substr($0, index($0, $4))
      printf "%s\t%s\t%s\t%d\n", b, $3, path, $2
    }' >> "$tmp"
done

echo "branch,file_path,size_mb,commit_sha,author" > "$out"
while IFS=$'\t' read -r br sha path size; do
  info=$(git -C "$mirror" log -n 1 --format='%H|%an' --find-object="$sha" "$br" 2>/dev/null || true)
  commit="${info%%|*}"
  author="${info#*|}"
  size_mb=$(awk -v s="$size" 'BEGIN {printf "%.2f", s/1024/1024}')
  path=${path//\"/\"\"}
  author=${author//\"/\"\"}
  echo "\"$br\",\"$path\",$size_mb,\"$commit\",\"$author\"" >> "$out"
done < "$tmp"

echo "Wrote: $out"
