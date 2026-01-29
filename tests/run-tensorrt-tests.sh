#!/bin/bash
#
# TensorRT-RTX Backend Test Runner
# Milestone 1 - Backend Infrastructure Testing
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}TensorRT-RTX Backend Test Suite - Milestone 1${NC}"
echo -e "${BLUE}===============================================${NC}"
echo

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found: $BUILD_DIR${NC}"
    echo "Please build the project first:"
    echo "  cmake -B build -DGGML_CUDA=ON -DGGML_TENSORRT=ON"
    echo "  cmake --build build -j8"
    exit 1
fi

# Check if library exists
if [ ! -f "$BUILD_DIR/lib/libggml-tensorrt.so" ] && [ ! -f "$BUILD_DIR/lib/libggml-tensorrt.dylib" ]; then
    echo -e "${RED}Error: TensorRT backend library not found${NC}"
    echo "Please build with: cmake -B build -DGGML_TENSORRT=ON"
    exit 1
fi

passed=0
failed=0
skipped=0

compile_test() {
    local test_name=$1
    local source_file=$2
    local output_file=$3

    echo -n "  Compiling $test_name... "

    g++ -o "$output_file" "$source_file" \
        -I "$PROJECT_ROOT/ggml/include" \
        -L "$BUILD_DIR/lib" \
        -lggml -lggml-tensorrt -lcudart \
        -Wl,-rpath,"$BUILD_DIR/lib" \
        > /tmp/compile_${test_name}.log 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Compilation output:"
        cat /tmp/compile_${test_name}.log | sed 's/^/    /'
        return 1
    fi
}

run_test() {
    local test_name=$1
    local test_cmd=$2

    echo -n "  Running $test_name... "

    if eval "$test_cmd" > /tmp/test_${test_name}.log 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((passed++))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Test output:"
        cat /tmp/test_${test_name}.log | sed 's/^/    /'
        ((failed++))
        return 1
    fi
}

# Build test programs
echo -e "${YELLOW}Building test programs...${NC}"
echo

mkdir -p "$SCRIPT_DIR/bin"

compile_test "test-tensorrt-init" \
    "$SCRIPT_DIR/test-tensorrt-init.cpp" \
    "$SCRIPT_DIR/bin/test-tensorrt-init" || ((failed++))

compile_test "test-tensorrt-devices" \
    "$SCRIPT_DIR/test-tensorrt-devices.cpp" \
    "$SCRIPT_DIR/bin/test-tensorrt-devices" || ((failed++))

compile_test "test-tensorrt-buffers" \
    "$SCRIPT_DIR/test-tensorrt-buffers.cpp" \
    "$SCRIPT_DIR/bin/test-tensorrt-buffers" || ((failed++))

compile_test "test-tensorrt-transfer" \
    "$SCRIPT_DIR/test-tensorrt-transfer.cpp" \
    "$SCRIPT_DIR/bin/test-tensorrt-transfer" || ((failed++))

if [ $failed -gt 0 ]; then
    echo
    echo -e "${RED}Some tests failed to compile. Aborting.${NC}"
    exit 1
fi

echo
echo -e "${YELLOW}Running tests...${NC}"
echo

# Phase 1: Backend Registration
echo -e "${BLUE}Phase 1: Backend Registration${NC}"
if [ -f "$SCRIPT_DIR/bin/test-tensorrt-init" ]; then
    run_test "backend-registration" "$SCRIPT_DIR/bin/test-tensorrt-init"
else
    echo -e "  ${YELLOW}SKIPPED${NC} (not compiled)"
    ((skipped++))
fi
echo

# Phase 2: Device Enumeration
echo -e "${BLUE}Phase 2: Device Enumeration${NC}"
if [ -f "$SCRIPT_DIR/bin/test-tensorrt-devices" ]; then
    run_test "device-enumeration" "$SCRIPT_DIR/bin/test-tensorrt-devices"
else
    echo -e "  ${YELLOW}SKIPPED${NC} (not compiled)"
    ((skipped++))
fi
echo

# Phase 3: Buffer Management
echo -e "${BLUE}Phase 3: Buffer Management${NC}"
if [ -f "$SCRIPT_DIR/bin/test-tensorrt-buffers" ]; then
    run_test "buffer-management" "$SCRIPT_DIR/bin/test-tensorrt-buffers"
else
    echo -e "  ${YELLOW}SKIPPED${NC} (not compiled)"
    ((skipped++))
fi
echo

# Phase 4: Tensor Transfer
echo -e "${BLUE}Phase 4: Tensor Transfer${NC}"
if [ -f "$SCRIPT_DIR/bin/test-tensorrt-transfer" ]; then
    run_test "tensor-transfer" "$SCRIPT_DIR/bin/test-tensorrt-transfer"
else
    echo -e "  ${YELLOW}SKIPPED${NC} (not compiled)"
    ((skipped++))
fi
echo

# Summary
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "Passed:  ${GREEN}$passed${NC}"
echo -e "Failed:  ${RED}$failed${NC}"
echo -e "Skipped: ${YELLOW}$skipped${NC}"
echo "Total:   $((passed + failed + skipped))"
echo

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All tests PASSED!${NC}"
    echo
    echo "Next steps:"
    echo "  1. The backend infrastructure is working correctly"
    echo "  2. You can now proceed with Milestone 2 (Core Operations)"
    echo "  3. Run 'llama-cli' to verify fallback to CUDA works"
    exit 0
else
    echo -e "${RED}✗ Some tests FAILED${NC}"
    echo
    echo "Please review the test output above and fix any issues."
    exit 1
fi
