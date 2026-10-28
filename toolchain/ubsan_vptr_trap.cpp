//===-- ubsan_vptr_trap.cpp - a trap-only -fsanitize=vptr handler ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The two symbols that -fsanitize=vptr code refers to, and nothing else, so a
// vptr check can be linked without the UBSan runtime.  Section 4c of
// toolchain/Dockerfile builds this into
// /opt/ubsan-vptr-trap/libclang_rt.ubsan_vptr_trap.a; ubsan-vptr-musl.cfg
// links it.
//
// The instrumented code hashes the object's vptr with the static type and
// looks the hash up in __ubsan_vptr_type_cache.  On a miss it calls the
// handler below.  The handler makes compiler-rt's decision
// (lib/ubsan/ubsan_type_hash_itanium.cpp, checkDynamicType and
// isDerivedFromAtOffset): it reads the type_info of the object's dynamic type
// from the vtable prefix and walks the Itanium base-class graph for the static
// type at the right offset.  As there, a virtual base on the path counts as a
// match.  A match fills the cache slot as compiler-rt does and returns.  A
// mismatch executes __builtin_trap(), as a -fsanitize-trap check does.  The
// handler prints nothing, makes no syscall, allocates nothing and has no
// global constructor.
//
// Two differences from compiler-rt.  It tests each address that the object's
// vptr leads it to, before it reads it, against the program's own image, where
// compiler-rt calls IsAccessibleMemoryRange, which writes the address to a
// pipe: this handler makes no syscall, and a statically linked program holds
// every vtable, type_info and type name it can reach inside its image, so the
// bound is exact here and asks the kernel nothing.  That covers the vtable
// prefix, each type_info node of the walk, the vptr of each node and the
// type_info and name string it leads to, and each node's whole base-class
// array.  Apart from the object's own storage, it reads no address outside the image, where reading could fault: at
// the vtable or its prefix such an address is a mismatch, and traps; at a node
// of the walk it ends that branch of the search, as a limit does (below).  Two
// reads are not tested: the object's own storage, which
// compiler-rt reads untested too, so an object whose own page is unmapped
// faults in either; and the check's static type_info, which the compiler emits
// into the image.  And it compares type_info objects by address
// or by the address of their names, which is exact in one statically linked
// image, where each type has one type_info.
//
// The walk has two limits that compiler-rt's has not.  It follows a path at
// most kMaxDepth (512) base classes down from the dynamic type, and it visits
// at most kMaxNodes (1 << 20) nodes in all, the dynamic type included.  The
// depth limit ends a base-class graph that loops inside the image, and the 513
// frames of a walk that deep fit well inside a musl thread's default 128 KiB
// stack.  The node limit ends a walk over an acyclic graph whose paths
// multiply, which the depth limit alone does not bound.
//
// A limit ends only the branch it stops: a node deeper than kMaxDepth, and every
// node after the kMaxNodes-th, is not visited, so nothing at or below it can
// match.  The walk goes on with the other bases of the nodes it has visited,
// and a virtual base among them still counts as a match, as it does in
// compiler-rt.  A search that ends with no match is a mismatch, and traps.  So
// a correct program whose match lies only past a limit traps, unless the walk
// counts a virtual base as a match: that takes a match more than 512 base
// classes down, or one that the walk would reach only after visiting more than
// 2^20 nodes, which takes about a million base subobjects of class type.
//
// It reads RTTI, so it is compiled with RTTI, and the code it checks must be
// too: clang emits no vptr check in a translation unit built with -fno-rtti.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <cstddef>
#include <typeinfo>

namespace __cxxabiv1 {
struct __class_type_info : std::type_info {};
struct __si_class_type_info : __class_type_info { const __class_type_info *__base_type; };
struct __base_class_type_info {
  const __class_type_info *__base_type;
  long __offset_flags;
  enum { __virtual_mask = 0x1, __offset_shift = 8 };
};
struct __vmi_class_type_info : __class_type_info {
  unsigned int __flags;
  unsigned int __base_count;
  __base_class_type_info __base_info[1];
};
}
namespace abi = __cxxabiv1;

typedef uintptr_t HashValue;
static const unsigned VptrTypeCacheSize = 128;
static const long VptrMaxOffsetToTop = 1 << 20;

// The ends of the program's image, from the linker.  Every vtable and every
// type_info of a static link lies between them.
extern "C" const char __executable_start[];
extern "C" const char _end[];

// The symbol the compiler emits the fast-path cache probe against.
extern "C" HashValue __ubsan_vptr_type_cache[VptrTypeCacheSize];
HashValue __ubsan_vptr_type_cache[VptrTypeCacheSize];

namespace {
// Whether an address the handler is about to read belongs to the image.  A
// freed object's stale vptr, and a wild one, point at the heap, at the stack or
// at nothing; each is outside the image, so the read does not happen and the
// handler traps instead of faulting.
bool in_image(const void *p, unsigned long size) {
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  uintptr_t lo = reinterpret_cast<uintptr_t>(__executable_start);
  uintptr_t hi = reinterpret_cast<uintptr_t>(_end);
  return a >= lo && a <= hi && hi - a >= size;
}
// Whether count elements of size elem, starting at p, all lie in the image.  A
// division, not a product, so that no count read from memory can overflow it.
bool in_image_array(const void *p, unsigned long count, unsigned long elem) {
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  uintptr_t lo = reinterpret_cast<uintptr_t>(__executable_start);
  uintptr_t hi = reinterpret_cast<uintptr_t>(_end);
  return a >= lo && a <= hi && count <= (hi - a) / elem;
}
// Mangled names of the three Itanium RTTI node kinds (libc++abi/libstdc++ share
// them).  Classifying by name sidesteps a cross-TU dynamic_cast, whose private
// type_info identity would not match libc++abi's own nodes.  Arrays, so that
// sizeof bounds how far strcmp can read the name it compares them with.
static const char kSI[]  = "N10__cxxabiv120__si_class_type_infoE";
static const char kVMI[] = "N10__cxxabiv121__vmi_class_type_infoE";
static_assert(sizeof(kVMI) >= sizeof(kSI), "the name bound below uses kVMI");
// The walk's two limits; the file comment above says what each costs.  The
// depth keeps the recursion inside a small thread stack, and ends a loop.  The
// node count ends a walk over an acyclic graph whose paths multiply, which the
// depth alone does not bound.
static const unsigned kMaxDepth = 512;
static const unsigned long kMaxNodes = 1ul << 20;
// checkTypeInfoEquality collapses to pointer identity in a single statically
// linked image (one copy of each type_info), which is our case.
bool typeinfo_eq(const std::type_info *a, const std::type_info *b) {
  return a == b || a->name() == b->name();
}
bool isDerivedFromAtOffset(const abi::__class_type_info *Derived,
                           const abi::__class_type_info *Base, long Offset,
                           unsigned Depth, unsigned long &Budget) {
  if (Depth > kMaxDepth || Budget == 0)
    return false;
  --Budget;
  // Every node of the walk is read, so every node must be in the image.
  if (!in_image(Derived, sizeof(abi::__vmi_class_type_info)))
    return false;
  if (typeinfo_eq(Derived, Base))
    return Offset == 0;
  // The node's kind is the name of its own dynamic type: its vptr, the
  // type_info pointer in front of that vtable, that type_info, and its name.
  // Each is tested before it is read.
  const void *NodeVptr = *reinterpret_cast<const void *const *>(Derived);
  if (!in_image(NodeVptr, 0))
    return false;
  const std::type_info *const *Slot =
      reinterpret_cast<const std::type_info *const *>(NodeVptr) - 1;
  if (!in_image(Slot, sizeof(*Slot)))
    return false;
  const std::type_info *NodeType = *Slot;
  if (!in_image(NodeType, sizeof(std::type_info)))
    return false;
  const char *k = NodeType->name();
  // strcmp stops at the first byte that differs, so it reads at most
  // sizeof(kVMI) bytes of k.
  if (!in_image(k, sizeof(kVMI)))
    return false;
  if (__builtin_strcmp(k, kSI) == 0)
    return isDerivedFromAtOffset(
        static_cast<const abi::__si_class_type_info *>(Derived)->__base_type,
        Base, Offset, Depth + 1, Budget);
  if (__builtin_strcmp(k, kVMI) != 0)
    return false;  // __class_type_info leaf (no bases) or an unknown node
  const abi::__vmi_class_type_info *VTI =
      static_cast<const abi::__vmi_class_type_info *>(Derived);
  // The whole base-class array must be in the image, however large the count.
  unsigned long Count = VTI->__base_count;
  if (!in_image_array(VTI->__base_info, Count,
                      sizeof(abi::__base_class_type_info)))
    return false;
  for (unsigned long base = 0; base != Count; ++base) {
    long OffsetHere = VTI->__base_info[base].__offset_flags >>
                      abi::__base_class_type_info::__offset_shift;
    if (VTI->__base_info[base].__offset_flags &
        abi::__base_class_type_info::__virtual_mask)
      return true;  // conservative: a virtual base cannot be ruled out here
    if (isDerivedFromAtOffset(VTI->__base_info[base].__base_type, Base,
                              Offset - OffsetHere, Depth + 1, Budget))
      return true;
  }
  return false;
}
struct VtablePrefix { long Offset; std::type_info *TypeInfo; };
bool checkDynamicType(void *Object, void *Type, HashValue Hash) {
  void *VtablePtr = *reinterpret_cast<void **>(Object);
  // The vtable, and the prefix in front of it, must be in the image.
  if (!in_image(VtablePtr, 0))
    return false;
  VtablePrefix *Prefix = reinterpret_cast<VtablePrefix *>(VtablePtr) - 1;
  if (!in_image(Prefix, sizeof(VtablePrefix)))
    return false;
  if (!Prefix->TypeInfo) return false;
  if (Prefix->Offset < -VptrMaxOffsetToTop || Prefix->Offset > VptrMaxOffsetToTop)
    return false;
  // Prefix->TypeInfo is a std::type_info*; every class node is layout-compatible
  // with __class_type_info, so treat it as one (its real kind is read by name in
  // isDerivedFromAtOffset).
  abi::__class_type_info *Derived =
      static_cast<abi::__class_type_info *>(Prefix->TypeInfo);
  abi::__class_type_info *Base = reinterpret_cast<abi::__class_type_info *>(Type);
  unsigned long Budget = kMaxNodes;
  if (!isDerivedFromAtOffset(Derived, Base, -Prefix->Offset, 0, Budget))
    return false;
  __ubsan_vptr_type_cache[Hash % VptrTypeCacheSize] = Hash;  // compiler-rt cache protocol
  return true;
}
// Must match compiler-rt ubsan_handlers_cxx.h: SourceLocation is 16 bytes
// (Filename ptr + Line + Column), then the TypeDescriptor ref, then TypeInfo.
struct SourceLocation { const char *Filename; uint32_t Line; uint32_t Column; };
struct DynamicTypeCacheMissData {
  SourceLocation Loc; const void *Type; void *TypeInfo; unsigned char TypeCheckKind;
};
} // namespace

extern "C" void __ubsan_handle_dynamic_type_cache_miss_abort(
    DynamicTypeCacheMissData *Data, uintptr_t Pointer, uintptr_t Hash) {
  if (checkDynamicType(reinterpret_cast<void *>(Pointer), Data->TypeInfo, Hash))
    return;                 // genuine cache miss: the type matched after all
  __builtin_trap();         // real mismatch -> SIGILL, like -fsanitize-trap would
}
// The non-abort variant is referenced when a vptr check is recoverable; route it
// to the same deterministic decision so either instrumentation form traps.
extern "C" void __ubsan_handle_dynamic_type_cache_miss(
    DynamicTypeCacheMissData *Data, uintptr_t Pointer, uintptr_t Hash) {
  if (checkDynamicType(reinterpret_cast<void *>(Pointer), Data->TypeInfo, Hash))
    return;
  __builtin_trap();
}
