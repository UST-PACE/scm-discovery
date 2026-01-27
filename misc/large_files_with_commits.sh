#!/usr/bin/env bash
set -euo pipefail

# ****************************************************************************************************************************
# DISCLAIMER: This script is provided "AS IS" without warranty of any kind.
# Use at your own risk. No liability for data loss, repo changes, or downtime.
# ****************************************************************************************************************************

# Dependencies: git, awk, and a POSIX shell (bash/zsh). For JSON metadata output, python3 is required.
# For private repos, SSH auth must be set up.

# What this bash script does:
#   Scans commits for large blobs in a GitLab repo mirror and outputs a CSV with branch, path, extension, size,
#   commit hash, and author. Also writes a JSON metadata report next to the CSV.
#   Use --default-branch or --branch to limit the scan. Use --per-commit to report per-commit file changes.

# Execute:
#   chmod +x large_files_with_commits.sh
#   ./large_files_with_commits.sh \
#     --url git@gitlab.com:group/project.git \
#     --threshold-mb 100 \
#     --mirror-root ~/.gl-ghec-mirror \
#     --out /tmp/large_files_with_commits.csv \
#     --default-branch

url=""
threshold_mb=""
mirror_root=""
out=""
meta_out=""
default_branch_only="false"
branch_name=""
per_commit="false"
threads="1"
summary_only="false"

usage() {
  cat <<'EOF'
Usage:
  ./large_files_with_commits.sh --url <gitlab_repo_url> --threshold-mb <mb> --mirror-root <dir> --out <csv> [--meta-out <json>] [--default-branch] [--branch <name>] [--per-commit] [--threads <n>] [--just-file-extension-count-maxsize]

Options:
  --url             GitLab repo SSH URL (required; https is converted to ssh)
  --threshold-mb    Size threshold in MB (required)
  --mirror-root     Mirror root directory (required)
  --out             Output CSV path (required)
  --meta-out        Output JSON metadata path (optional; default is <out>.json)
  --default-branch  Scan only the default branch (HEAD).
  --branch          Scan only the named branch.
  --per-commit      Report files >= threshold per commit (slower, more detailed).
  --threads         Parallel threads for --per-commit (default 1).
  --just-file-extension-count-maxsize  Output summary by file extension and max size.
  -h, --help        Show this help
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
    --meta-out)
      meta_out="$2"
      shift 2
      ;;
    --default-branch)
      default_branch_only="true"
      shift 1
      ;;
    --branch)
      branch_name="$2"
      shift 2
      ;;
    --per-commit)
      per_commit="true"
      shift 1
      ;;
    --threads)
      threads="$2"
      shift 2
      ;;
    --just-file-extension-count-maxsize)
      summary_only="true"
      shift 1
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
if [[ "$default_branch_only" == "true" && -n "$branch_name" ]]; then
  echo "Error: --default-branch and --branch cannot be used together."
  exit 1
fi
if ! [[ "$threads" =~ ^[0-9]+$ ]] || [[ "$threads" -lt 1 ]]; then
  echo "Error: --threads must be a positive integer."
  exit 1
fi
if [[ "$summary_only" == "true" && "$per_commit" != "true" ]]; then
  echo "Note: --just-file-extension-count-maxsize works best with --per-commit; continuing."
fi

if [[ -z "$meta_out" ]]; then
  if [[ "$out" == *.csv ]]; then
    meta_out="${out%.csv}.json"
  else
    meta_out="${out}.json"
  fi
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required to write JSON metadata."
  exit 1
fi

threshold_bytes=$((threshold_mb * 1024 * 1024))

raw_url="$url"
if [[ "$url" == http://* || "$url" == https://* ]]; then
  tmp="${url#*://}"
  host="${tmp%%/*}"
  path="${tmp#*/}"
elif [[ "$url" == ssh://* ]]; then
  tmp="${url#ssh://}"
  tmp="${tmp#git@}"
  host="${tmp%%/*}"
  path="${tmp#*/}"
elif [[ "$url" == git@*:* ]]; then
  host="${url#git@}"; host="${host%%:*}"
  path="${url#*:}"
else
  echo "Error: unrecognized URL format: $url"
  exit 1
fi

path="${path%.git}"
url="git@${host}:${path}.git"
mirror="${mirror_root}/${host}/${path}.git"
mkdir -p "$(dirname "$mirror")"

echo "Repo URL: $raw_url"
echo "SSH URL: $url"
echo "Mirror: $mirror"
echo "Threshold: ${threshold_mb} MB"
if [[ "$default_branch_only" == "true" ]]; then
  echo "Default branch only: true"
elif [[ -n "$branch_name" ]]; then
  echo "Branch only: $branch_name"
fi
echo "Per-commit mode: $per_commit"
if [[ "$per_commit" == "true" ]]; then
  echo "Threads: $threads"
fi
if [[ "$summary_only" == "true" ]]; then
  echo "Summary only: true"
fi

if [[ ! -d "$mirror" ]]; then
  echo "Cloning mirror..."
  git clone --mirror "$url" "$mirror"
fi

tmp_all="$(mktemp)"
branches_file="$(mktemp)"
count_objects_file="$(mktemp)"
trap 'rm -f "$tmp_all" "$branches_file" "$count_objects_file"' EXIT

branches=()
if [[ "$default_branch_only" == "true" ]]; then
  head_branch=$(git -C "$mirror" symbolic-ref -q --short HEAD || true)
  if [[ -z "$head_branch" ]]; then
    head_branch=$(git -C "$mirror" for-each-ref --format='%(refname:short)' refs/heads | head -n 1)
  fi
  if [[ -z "$head_branch" ]]; then
    echo "Error: Unable to determine default branch."
    exit 1
  fi
  branches=("$head_branch")
elif [[ -n "$branch_name" ]]; then
  if ! git -C "$mirror" show-ref --verify --quiet "refs/heads/$branch_name"; then
    echo "Error: Branch not found: $branch_name"
    exit 1
  fi
  branches=("$branch_name")
else
  mapfile -t branches < <(git -C "$mirror" for-each-ref --format='%(refname:short)' refs/heads)
fi

echo "Branches found: ${#branches[@]}"
branch_total=${#branches[@]}
branch_index=0
if [[ "$summary_only" == "true" ]]; then
  echo "branch,file_type,file_count,commit_count,max_size_mb" > "$out"
fi
printf "%s\n" "${branches[@]}" > "$branches_file"

branch_count=$(git -C "$mirror" for-each-ref --format='%(refname:short)' refs/heads | wc -l | tr -d ' ')
tag_count=$(git -C "$mirror" for-each-ref --format='%(refname:short)' refs/tags | wc -l | tr -d ' ')
default_branch=$(git -C "$mirror" symbolic-ref -q --short HEAD || true)
commit_count=$(git -C "$mirror" rev-list --all --count)

repo_disk_size_bytes=""
if du -sb "$mirror" >/dev/null 2>&1; then
  repo_disk_size_bytes=$(du -sb "$mirror" | awk '{print $1}')
else
  repo_disk_size_bytes=$(du -sk "$mirror" | awk '{print $1 * 1024}')
fi
git -C "$mirror" count-objects -v > "$count_objects_file" || true

for br in "${branches[@]}"; do
  branch_index=$((branch_index + 1))
  echo "Scanning branch ($branch_index/$branch_total): $br"
  branch_tmp="$(mktemp)"
  if [[ "$per_commit" == "true" ]]; then
    if [[ "$threads" -gt 1 ]] && command -v xargs >/dev/null 2>&1; then
      lockfile="$(mktemp)"
      export MIRROR_PATH="$mirror"
      export THRESHOLD_BYTES="$threshold_bytes"
      export TMP_OUT="$branch_tmp"
      export LOCKFILE="$lockfile"
      export BRANCH_NAME="$br"
      export USE_FLOCK="false"
      if command -v flock >/dev/null 2>&1; then
        export USE_FLOCK="true"
      fi
      git -C "$mirror" rev-list "$br" | xargs -P "$threads" -I {} bash -c '
        commit="$1"
        mirror="$MIRROR_PATH"
        branch="$BRANCH_NAME"
        threshold="$THRESHOLD_BYTES"
        tmp="$TMP_OUT"
        lock="$LOCKFILE"
        info=$(git -C "$mirror" show -s --format="%H|%an|%ad" --date=short "$commit")
        commit_sha="${info%%|*}"
        rest="${info#*|}"
        author="${rest%%|*}"
        commit_date="${rest#*|}"
        git -C "$mirror" diff-tree -r --no-commit-id --raw -z --diff-filter=AM "$commit" | \
        while IFS= read -r -d "" meta && IFS= read -r -d "" path; do
          meta="${meta#:}"
          set -- $meta
          newsha="${4:-}"
          if [[ -z "$newsha" ]]; then
            continue
          fi
          obj_type=$(git -C "$mirror" cat-file -t "$newsha" 2>/dev/null || true)
          if [[ "$obj_type" != "blob" ]]; then
            continue
          fi
          size=$(git -C "$mirror" cat-file -s "$newsha" 2>/dev/null || echo "0")
          if [[ "$size" -ge "$threshold" ]]; then
            if [[ "$USE_FLOCK" == "true" ]]; then
              {
                flock 9
                printf "%s\t%s\t%s\t%s\t%s\t%d\n" "$branch" "$commit_sha" "$author" "$commit_date" "$path" "$size" >> "$tmp"
              } 9>>"$lock"
            else
              printf "%s\t%s\t%s\t%s\t%s\t%d\n" "$branch" "$commit_sha" "$author" "$commit_date" "$path" "$size" >> "$tmp"
            fi
          fi
        done
      ' _ {}
      rm -f "$lockfile"
    else
      git -C "$mirror" rev-list "$br" | while read -r commit; do
        info=$(git -C "$mirror" show -s --format='%H|%an|%ad' --date=short "$commit")
        commit_sha="${info%%|*}"
        rest="${info#*|}"
        author="${rest%%|*}"
        commit_date="${rest#*|}"
        git -C "$mirror" diff-tree -r --no-commit-id --raw -z --diff-filter=AM "$commit" | \
        while IFS= read -r -d '' meta && IFS= read -r -d '' path; do
          meta="${meta#:}"
          set -- $meta
          newsha="${4:-}"
          if [[ -z "$newsha" ]]; then
            continue
          fi
          obj_type=$(git -C "$mirror" cat-file -t "$newsha" 2>/dev/null || true)
          if [[ "$obj_type" != "blob" ]]; then
            continue
          fi
          size=$(git -C "$mirror" cat-file -s "$newsha" 2>/dev/null || echo "0")
          if [[ "$size" -ge "$threshold_bytes" ]]; then
            printf "%s\t%s\t%s\t%s\t%s\t%d\n" "$br" "$commit_sha" "$author" "$commit_date" "$path" "$size" >> "$branch_tmp"
          fi
        done
      done
    fi
  else
    git -C "$mirror" rev-list --objects "$br" | \
    git -C "$mirror" cat-file --batch-check='%(objecttype) %(objectsize) %(objectname) %(rest)' | \
    awk -v b="$br" -v t="$threshold_bytes" '
      $1=="blob" && $2>=t {
        path = substr($0, index($0, $4))
        printf "%s\t%s\t%s\t%d\n", b, $3, path, $2
      }' >> "$branch_tmp"
  fi

  branch_lines=0
  if [[ -s "$branch_tmp" ]]; then
    branch_lines=$(wc -l < "$branch_tmp" | tr -d ' ')
    cat "$branch_tmp" >> "$tmp_all"
    if [[ "$summary_only" == "true" ]]; then
      if [[ "$per_commit" == "true" ]]; then
        awk -F'\t' '
          {
            br=$1; commit=$2; path=$5; size=$6;
            base=path; sub(/^.*\//,"",base);
            ext="(none)";
            if (base ~ /\./ && base !~ /^\./) { ext="." substr(base, index(base,".")+1) }
            key=br SUBSEP ext;
            keyc=br SUBSEP ext SUBSEP commit;
            count[key]++; if (size>max[key]) max[key]=size;
            if (!(seen[keyc]++)) commit_count[key]++;
          }
          END {
            for (k in count) {
              split(k,a,SUBSEP);
              printf "\"%s\",\"%s\",%d,%d,%.2f\n", a[1], a[2], count[k], commit_count[k], max[k]/1024/1024
            }
          }
        ' "$branch_tmp" >> "$out"
        awk -F'\t' '
          BEGIN {
            topn=3; idx=0;
          }
          {
            path=$5; size=$6;
            base=path; sub(/^.*\//,"",base);
            ext="(none)";
            if (base ~ /\./ && base !~ /^\./) { ext="." substr(base, index(base,".")+1) }
            files[ext]=files[ext] + 1;
            if (size > max_size) { max_size=size; max_ext=ext; max_path=path; }
            if (size > top_size[1]) {
              top_size[3]=top_size[2]; top_ext[3]=top_ext[2]; top_path[3]=top_path[2];
              top_size[2]=top_size[1]; top_ext[2]=top_ext[1]; top_path[2]=top_path[1];
              top_size[1]=size; top_ext[1]=ext; top_path[1]=path;
            } else if (size > top_size[2]) {
              top_size[3]=top_size[2]; top_ext[3]=top_ext[2]; top_path[3]=top_path[2];
              top_size[2]=size; top_ext[2]=ext; top_path[2]=path;
            } else if (size > top_size[3]) {
              top_size[3]=size; top_ext[3]=ext; top_path[3]=path;
            }
          }
          END {
            if (max_size > 0) {
              printf "  Largest in branch: %s (%.2f MB) %s\n", max_ext, max_size/1024/1024, max_path;
            }
            for (i=1; i<=3; i++) {
              if (top_size[i] > 0) {
                printf "  Top %d: %s (%.2f MB) %s\n", i, top_ext[i], top_size[i]/1024/1024, top_path[i];
              }
            }
          }
        ' "$branch_tmp"
      else
        awk -F'\t' '
          {
            br=$1; path=$3; size=$4;
            base=path; sub(/^.*\//,"",base);
            ext="(none)";
            if (base ~ /\./ && base !~ /^\./) { ext="." substr(base, index(base,".")+1) }
            key=br SUBSEP ext;
            count[key]++; if (size>max[key]) max[key]=size;
          }
          END {
            for (k in count) {
              split(k,a,SUBSEP);
              printf "\"%s\",\"%s\",%d,%d,%.2f\n", a[1], a[2], count[k], 0, max[k]/1024/1024
            }
          }
        ' "$branch_tmp" >> "$out"
        awk -F'\t' '
          {
            path=$3; size=$4;
            base=path; sub(/^.*\//,"",base);
            ext="(none)";
            if (base ~ /\./ && base !~ /^\./) { ext="." substr(base, index(base,".")+1) }
            if (size > max_size) { max_size=size; max_ext=ext; max_path=path; }
            if (size > top_size[1]) {
              top_size[3]=top_size[2]; top_ext[3]=top_ext[2]; top_path[3]=top_path[2];
              top_size[2]=top_size[1]; top_ext[2]=top_ext[1]; top_path[2]=top_path[1];
              top_size[1]=size; top_ext[1]=ext; top_path[1]=path;
            } else if (size > top_size[2]) {
              top_size[3]=top_size[2]; top_ext[3]=top_ext[2]; top_path[3]=top_path[2];
              top_size[2]=size; top_ext[2]=ext; top_path[2]=path;
            } else if (size > top_size[3]) {
              top_size[3]=size; top_ext[3]=ext; top_path[3]=path;
            }
          }
          END {
            if (max_size > 0) {
              printf "  Largest in branch: %s (%.2f MB) %s\n", max_ext, max_size/1024/1024, max_path;
            }
            for (i=1; i<=3; i++) {
              if (top_size[i] > 0) {
                printf "  Top %d: %s (%.2f MB) %s\n", i, top_ext[i], top_size[i]/1024/1024, top_path[i];
              }
            }
          }
        ' "$branch_tmp"
      fi
    fi
  fi
  echo "Branch matches: $branch_lines"
  rm -f "$branch_tmp"
done

large_count=0
max_size_bytes=0
max_branch=""
max_path=""
max_commit=""
max_author=""
max_commit_date=""

if [[ "$summary_only" == "true" ]]; then
  meta_tmp="$(mktemp)"
  if [[ "$per_commit" == "true" ]]; then
    awk -F'\t' -v meta="$meta_tmp" '
      {
        br=$1; commit=$2; author=$3; cdate=$4; path=$5; size=$6;
        base=path; sub(/^.*\//,"",base);
        ext="(none)";
        if (base ~ /\./ && base !~ /^\./) { ext="." substr(base, index(base,".")+1) }
        key=br SUBSEP ext;
        keyc=br SUBSEP ext SUBSEP commit;
        count[key]++; if (size>max[key]) max[key]=size;
        if (!(seen[keyc]++)) commit_count[key]++;
        total++; if (size>max_size) { max_size=size; max_branch=br; max_path=path; max_commit=commit; max_author=author; max_date=cdate }
      }
      END {
        printf "%d\t%d\t%s\t%s\t%s\t%s\t%s\n", total, max_size, max_branch, max_path, max_commit, max_author, max_date > meta
      }
    ' "$tmp_all"
  else
    awk -F'\t' -v meta="$meta_tmp" '
      {
        br=$1; sha=$2; path=$3; size=$4;
        base=path; sub(/^.*\//,"",base);
        ext="(none)";
        if (base ~ /\./ && base !~ /^\./) { ext="." substr(base, index(base,".")+1) }
        key=br SUBSEP ext;
        count[key]++; if (size>max[key]) max[key]=size;
        total++; if (size>max_size) { max_size=size; max_branch=br; max_path=path; max_commit=sha }
      }
      END {
        printf "%d\t%d\t%s\t%s\t%s\t%s\t%s\n", total, max_size, max_branch, max_path, max_commit, "", "" > meta
      }
    ' "$tmp_all"
  fi
  large_count=$(cut -f1 "$meta_tmp" 2>/dev/null || echo "0")
  max_size_bytes=$(cut -f2 "$meta_tmp" 2>/dev/null || echo "0")
  max_branch=$(cut -f3 "$meta_tmp" 2>/dev/null || echo "")
  max_path=$(cut -f4 "$meta_tmp" 2>/dev/null || echo "")
  max_commit=$(cut -f5 "$meta_tmp" 2>/dev/null || echo "")
  max_author=$(cut -f6 "$meta_tmp" 2>/dev/null || echo "")
  max_commit_date=$(cut -f7 "$meta_tmp" 2>/dev/null || echo "")
  rm -f "$meta_tmp"
else
  if [[ "$per_commit" == "true" ]]; then
    echo "branch,file_path,extension,size_mb,commit_sha,author,commit_date" > "$out"
    while IFS=$'\t' read -r br commit author commit_date path size; do
      size_mb=$(awk -v s="$size" 'BEGIN {printf "%.2f", s/1024/1024}')
      base="${path##*/}"
      ext=""
      if [[ "$base" == *.* && "$base" != .* ]]; then
        ext="${base##*.}"
      fi
      path=${path//\"/\"\"}
      author=${author//\"/\"\"}
      echo "\"$br\",\"$path\",\"$ext\",$size_mb,\"$commit\",\"$author\",\"$commit_date\"" >> "$out"
      large_count=$((large_count + 1))
      if [[ "$size" -gt "$max_size_bytes" ]]; then
        max_size_bytes="$size"
        max_branch="$br"
        max_path="$path"
        max_commit="$commit"
        max_author="$author"
        max_commit_date="$commit_date"
      fi
    done < "$tmp_all"
  else
    echo "branch,file_path,extension,size_mb,commit_sha,author" > "$out"
    while IFS=$'\t' read -r br sha path size; do
      info=$(git -C "$mirror" log -n 1 --format='%H|%an' --find-object="$sha" "$br" 2>/dev/null || true)
      commit="${info%%|*}"
      author="${info#*|}"
      size_mb=$(awk -v s="$size" 'BEGIN {printf "%.2f", s/1024/1024}')
      base="${path##*/}"
      ext=""
      if [[ "$base" == *.* && "$base" != .* ]]; then
        ext="${base##*.}"
      fi
      path=${path//\"/\"\"}
      author=${author//\"/\"\"}
      echo "\"$br\",\"$path\",\"$ext\",$size_mb,\"$commit\",\"$author\"" >> "$out"
      large_count=$((large_count + 1))
      if [[ "$size" -gt "$max_size_bytes" ]]; then
        max_size_bytes="$size"
        max_branch="$br"
        max_path="$path"
        max_commit="$commit"
        max_author="$author"
      fi
    done < "$tmp_all"
  fi
fi

row_count=$((large_count))

export META_INPUT_URL="$raw_url"
export META_SSH_URL="$url"
export META_MIRROR_PATH="$mirror"
export META_THRESHOLD_MB="$threshold_mb"
export META_THRESHOLD_BYTES="$threshold_bytes"
export META_DEFAULT_BRANCH="$default_branch"
export META_BRANCH_COUNT="$branch_count"
export META_TAG_COUNT="$tag_count"
export META_COMMIT_COUNT="$commit_count"
export META_REPO_DISK_SIZE_BYTES="$repo_disk_size_bytes"
export META_LARGE_COUNT="$large_count"
export META_MAX_SIZE_BYTES="$max_size_bytes"
export META_MAX_BRANCH="$max_branch"
export META_MAX_PATH="$max_path"
export META_MAX_COMMIT="$max_commit"
export META_MAX_AUTHOR="$max_author"
export META_MAX_COMMIT_DATE="$max_commit_date"
export META_MODE="$per_commit"

python3 - "$meta_out" "$branches_file" "$count_objects_file" <<'PY'
import json
import os
import sys

def parse_count_objects(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result

def env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

def env_float(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default

meta_out = sys.argv[1]
branches_file = sys.argv[2]
count_objects_file = sys.argv[3]

with open(branches_file, "r", encoding="utf-8") as fh:
    branches = [line.strip() for line in fh if line.strip()]

try:
    with open(count_objects_file, "r", encoding="utf-8") as fh:
        count_objects_raw = fh.read()
except FileNotFoundError:
    count_objects_raw = ""

max_size_bytes = env_int("META_MAX_SIZE_BYTES", 0) or 0

meta = {
    "input_url": os.environ.get("META_INPUT_URL"),
    "ssh_url": os.environ.get("META_SSH_URL"),
    "mirror_path": os.environ.get("META_MIRROR_PATH"),
    "threshold_mb": env_float("META_THRESHOLD_MB"),
    "threshold_bytes": env_int("META_THRESHOLD_BYTES"),
    "default_branch": os.environ.get("META_DEFAULT_BRANCH") or None,
    "branch_count": env_int("META_BRANCH_COUNT"),
    "tag_count": env_int("META_TAG_COUNT"),
    "commit_count": env_int("META_COMMIT_COUNT"),
    "repo_disk_size_bytes": env_int("META_REPO_DISK_SIZE_BYTES"),
    "branches_scanned": len(branches),
    "branches_scanned_list": branches,
    "mode": "per-commit" if os.environ.get("META_MODE") == "true" else "unique-blob",
    "large_files": {
        "count": env_int("META_LARGE_COUNT", 0) or 0,
        "max_size_bytes": max_size_bytes,
        "max_size_mb": round(max_size_bytes / 1024 / 1024, 2) if max_size_bytes else 0,
        "max_branch": os.environ.get("META_MAX_BRANCH") or None,
        "max_path": os.environ.get("META_MAX_PATH") or None,
        "max_commit": os.environ.get("META_MAX_COMMIT") or None,
        "max_author": os.environ.get("META_MAX_AUTHOR") or None,
        "max_commit_date": os.environ.get("META_MAX_COMMIT_DATE") or None,
    },
    "git_count_objects": parse_count_objects(count_objects_raw),
}

with open(meta_out, "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2, sort_keys=True)
PY

echo "Rows written: $row_count"
if [[ "$row_count" -eq 0 ]]; then
  echo "No files >= ${threshold_mb} MB found in the scanned branches."
fi
echo "Wrote: $out"
echo "Wrote metadata: $meta_out"
