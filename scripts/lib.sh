# Shared helpers for the run scripts (source this file).

# Next free run name for a base: BASE, then BASE-r2, BASE-r3, ... so no run overwrites another.
# The name is taken if out/<folder>/<prefix>_<name>.csv exists for any given
# <folder>/<prefix>, e.g. peaks/peaks or benchmark/bench.
next_run() {
  local base=$1
  shift
  local n=1 name=$base prefix taken
  while :; do
    taken=0
    for prefix in "$@"; do
      [ -e "out/${prefix}_${name}.csv" ] && taken=1
    done
    [ "$taken" = 0 ] && break
    n=$((n + 1))
    name="$base-r$n"
  done
  echo "$name"
}

# Write a command's output to a file only if the file does not exist yet.
write_once() {
  local out=$1
  shift
  [ -e "$out" ] || "$@" >"$out"
}

# Move fetched files (<src>/<folder>/<file>) into out/<folder>/, refusing to replace
# any that exist.
move_new() {
  local src=$1 f dest
  for f in "$src"/*/*; do
    [ -e "$f" ] || continue
    dest="out/$(basename "$(dirname "$f")")/$(basename "$f")"
    if [ -e "$dest" ]; then
      echo "error: $dest exists; not overwriting" >&2
      return 1
    fi
    mkdir -p "$(dirname "$dest")"
    mv "$f" "$dest"
  done
}

# Settings that SMOKE=1 passes to every run (empty for a real run).
RUN_OVERRIDES=${RUN_OVERRIDES:-}
