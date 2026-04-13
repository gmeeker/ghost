// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <ghost/argument_buffer.h>
#include <ghost/attribute.h>
#include <ghost/device.h>
#include <ghost/implementation/impl_device.h>

namespace ghost {

// All special members are defined out-of-line because the strong-ref members
// (shared_ptr<implementation::Buffer>, shared_ptr<implementation::Image>,
// shared_ptr<ArgumentBuffer>) reference types that are forward-declared in
// attribute.h. Defining the destructor and copy/move ops here ensures the
// shared_ptr destructors are emitted in a TU where the pointee types are
// complete.

Attribute::Attribute() : _type(Type_Unknown), _count(0) {}

Attribute::~Attribute() = default;
Attribute::Attribute(const Attribute&) = default;
Attribute::Attribute(Attribute&&) noexcept = default;
Attribute& Attribute::operator=(const Attribute&) = default;
Attribute& Attribute::operator=(Attribute&&) noexcept = default;

Attribute::Attribute(char* s) : _type(Type_String), _count(1), _s(s) {}

Attribute::Attribute(const char* s) : _type(Type_String), _count(1), _s(s) {}

Attribute::Attribute(const std::string& s)
    : _type(Type_String), _count(1), _s(s) {}

Attribute::Attribute(Buffer* b) : _type(Type_Buffer), _count(1) {
  if (b) _bufferImpl = b->impl();
}

Attribute::Attribute(Buffer& b)
    : _type(Type_Buffer), _count(1), _bufferImpl(b.impl()) {}

Attribute::Attribute(Image* i) : _type(Type_Image), _count(1) {
  if (i) _imageImpl = i->impl();
}

Attribute::Attribute(Image& i)
    : _type(Type_Image), _count(1), _imageImpl(i.impl()) {}

Attribute::Attribute(Image& i, const SamplerDescription& sampler)
    : _type(Type_Image), _count(1), _imageImpl(i.impl()), _sampler(sampler) {}

Attribute::Attribute(ArgumentBuffer* ab)
    : _type(Type_ArgumentBuffer), _count(1) {
  if (ab) _argBuffer = std::make_shared<ArgumentBuffer>(*ab);
}

Attribute::Attribute(ArgumentBuffer& ab)
    : _type(Type_ArgumentBuffer),
      _count(1),
      _argBuffer(std::make_shared<ArgumentBuffer>(ab)) {}

}  // namespace ghost
