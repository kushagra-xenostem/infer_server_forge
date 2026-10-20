#!/bin/bash
# Print the hardware/software facts this repo's docs rely on. No secrets: network identifiers are not printed.
# Usage: bash tools/collect_env.sh > docs/environment.txt
ROCK=${ROCK:-/data/rocm/therock-7.14/install}
LLAMA=${LLAMA:-/data/projects/llama.cpp}
sec() { echo; echo "## $*"; }

sec "OS"
grep -E '^(PRETTY_NAME|VERSION_CODENAME)=' /etc/os-release; echo "kernel: $(uname -r)"
sec "Motherboard / BIOS"
echo "$(cat /sys/class/dmi/id/board_vendor) $(cat /sys/class/dmi/id/board_name) | BIOS $(cat /sys/class/dmi/id/bios_version) $(cat /sys/class/dmi/id/bios_date)"
sec "CPU / RAM"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|CPU max MHz'
grep MemTotal /proc/meminfo; grep SwapTotal /proc/meminfo
sec "Disks"
lsblk -d -o NAME,SIZE,ROTA,MODEL | grep -v loop
sec "GPUs"
lspci -nn | grep -iE 'vga|display'
GPU=$(lspci -D | awk '/VGA.*Navi/{print $1; exit}')
[ -n "$GPU" ] && echo "RX 6700 subsystem vendor: $(cat /sys/bus/pci/devices/$GPU/subsystem_vendor)"
P=$(readlink -f /sys/bus/pci/devices/$GPU 2>/dev/null); d=$P
while [ -n "$d" ] && [ "$d" != /sys/devices ]; do
  [ -f $d/current_link_width ] && echo "PCIe $(basename $d): $(cat $d/current_link_speed) x$(cat $d/current_link_width)"; d=$(dirname $d); done
echo "amdgpu module: $(modinfo -F filename amdgpu)"
dpkg -l linux-firmware 2>/dev/null | awk '/^ii/{print "linux-firmware " $3}'
sec "ROCm (TheRock)"
export PATH=$ROCK/bin:$ROCK/lib/llvm/bin:$PATH LD_LIBRARY_PATH=$ROCK/lib:$ROCK/lib/llvm/lib
echo "HIP: $(hipconfig --version 2>/dev/null)"; clang++ --version | head -1
amd-smi static --asic -g 0 2>/dev/null | grep -E 'MARKET_NAME|TARGET_GRAPHICS_VERSION'
sec "llama.cpp"
$LLAMA/build/bin/llama-server --version 2>&1 | grep -E 'version|built'
grep -E '^(AMDGPU_TARGETS|GGML_HIP|CMAKE_BUILD_TYPE|CMAKE_HIP_COMPILER)[:=]' $LLAMA/build/CMakeCache.txt
ldd $LLAMA/build/bin/llama-server | grep -E 'amdhip|hipblas|rocblas' | sed 's/ (0x.*//'
sec "Models"
for f in $HOME/models/gguf/*.gguf; do echo "$(basename $f)  $(stat -c %s $f) bytes"; done
sec "Tooling"
python3 --version; docker --version 2>/dev/null; echo "tailscale $(tailscale version 2>/dev/null | head -1)"
ls $HOME/.vscode-server/extensions 2>/dev/null | grep -E 'claude-dev' | sed 's/^/cline extension: /'
