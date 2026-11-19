// The Python test supplies the actual lifted kernel and the cluster launch shape.
#include "lift_host.h"

static bool check_query() {
  uint8_t bank[LIFT_CBANK] = {};
  for (uint32_t tag : {0u, 1u, 0xfffffffeu, 0xffffffffu}) {
    uint32_t words[] = {0x89abcdefu, 0x9abc1234u, tag, 0xa5a5a5a5, 91, 92, 93, 94};
    uint32_t expected[8];
    memcpy(expected, words, sizeof(words));
    expected[0] = tag & 1;
    if (expected[0]) {
      expected[4] = words[0];
      expected[5] = 0x1234;
      expected[6] = 0x9abc;
    }
    void *buf = lift_dev(words, sizeof(words));
    lift_put<void *>(bank, 896, buf);
    lift_launch("probe", dim3(1), dim3(1), 0, "896:8 904:4 908:4", bank);
    LIFT_CHECK(cudaMemcpy(words, buf, sizeof(words), cudaMemcpyDeviceToHost));
    LIFT_CHECK(cudaFree(buf));
    for (unsigned i = 0; i < 8; ++i) {
      if (words[i] != expected[i]) {
        printf("query tag=%x word=%u got=%x expected=%x\n", tag, i, words[i], expected[i]);
        return false;
      }
    }
  }
  return true;
}

static bool check_pair() {
  uint8_t bank[LIFT_CBANK] = {};
  uint32_t words[66];
  void *buf = lift_dev(nullptr, sizeof(words));
  lift_put<uint32_t *>(bank, 896, static_cast<uint32_t *>(buf) + 1);
  lift_put<uint32_t>(bank, 904, 32);
  for (unsigned trial = 1; trial <= 2; ++trial) {
    const uint32_t guard = trial * 0x11111111u;
    for (auto &word : words) word = guard;
    LIFT_CHECK(cudaMemcpy(buf, words, sizeof(words), cudaMemcpyHostToDevice));
    lift_launch("probe", dim3(2), dim3(32), 0, "896:8 904:4 908:4", bank);
    LIFT_CHECK(cudaMemcpy(words, buf, sizeof(words), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < 66; ++i) {
      // A full 512-column allocation owns all TMEM and starts at column zero.
      const uint32_t expected = (i == 0 || i == 65) ? guard : 0;
      if (words[i] != expected) {
        printf("pair trial=%u word=%u got=%x expected=%x\n", trial, i, words[i], expected);
        return false;
      }
    }
  }
  LIFT_CHECK(cudaFree(buf));
  return true;
}

int main(int argc, char **argv) {
  if (argc != 2) return 1;
  bool ok = strcmp(argv[1], "pair") == 0 ? check_pair() : check_query();
  if (ok) puts("PASSED CPU comparison");
  return ok ? 0 : 1;
}
