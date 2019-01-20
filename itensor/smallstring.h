//
// Distributed under the ITensor Library License, Version 1.2
//    (See accompanying LICENSE file.)
//
#ifndef __ITENSOR_SMALLSTRING_H_
#define __ITENSOR_SMALLSTRING_H_

#include <array>
#include <cstring>
#include <cctype>
#include <iostream>
#include "itensor/util/error.h"
#include "itensor/util/print.h"

#ifdef DEBUG
#define CHECK_IND(X) check_ind(X);
#else
#define CHECK_IND(X)
#endif

namespace itensor {

size_t inline constexpr 
SmallStringSize() { return 7ul; }

size_t inline constexpr 
SmallStringStoreSize() { return 1+SmallStringSize(); }

struct SmallString
    {
    using storage_type = std::array<char,SmallStringStoreSize()>;
    private:
    storage_type name_;
    public:

    SmallString();

    SmallString(const char* name);

    SmallString(std::string const& name) : SmallString(name.c_str()) { }

    size_t static constexpr
    size() { return SmallStringSize(); }

    const char*
    c_str() const { assert(name_[size()]=='\0'); return &(name_[0]); }

    operator const char*() const { return c_str(); }

    const char&
    operator[](size_t i) const { CHECK_IND(i) return name_[i]; }

    char&
    operator[](size_t i) { CHECK_IND(i) return name_[i]; }

    void
    set(size_t i, const char c) { CHECK_IND(i) name_[i] = c; return; }

    explicit
    operator const int64_t() const { return reinterpret_cast<const int64_t&>(name_[0]); }

    private:
    void
    check_ind(size_t j) const
        {
        if(j >= size()) throw std::runtime_error("SmallString: index out of range");
        }
    };


bool inline
operator==(SmallString const& t1, SmallString const& t2)
    {
    for(size_t j = 0; j < SmallString::size(); ++j)
        if(t1[j] != t2[j]) return false;
    return true;
    }

bool inline
operator!=(SmallString const& t1, SmallString const& t2)
    {
    return !operator==(t1,t2);
    }

bool inline
operator<(SmallString const& t1, SmallString const& t2)
    {
    return int64_t(t1) < int64_t(t2);
    }

bool inline
operator>(SmallString const& t1, SmallString const& t2)
    {
    return t2 < t1;
    }

bool inline
operator<=(SmallString const& t1, SmallString const& t2)
    {
    return int64_t(t1) <= int64_t(t2);
    }

bool inline
operator>=(SmallString const& t1, SmallString const& t2)
    {
    return t2 <= t1;
    }

bool inline
operator==(SmallString const& t1, std::string s2)
    {
    return operator==(t1,SmallString(s2));
    }
bool inline
operator==(std::string s1, SmallString const& t2)
    {
    return operator==(SmallString(s1),t2);
    }
bool inline
operator!=(SmallString const& t1, std::string s2)
    {
    return operator!=(t1,SmallString(s2));
    }
bool inline
operator!=(std::string s1, SmallString const& t2)
    {
    return operator!=(SmallString(s1),t2);
    }

bool inline
operator==(SmallString const& t1, const char* s2)
    {
    return operator==(t1,SmallString(s2));
    }
bool inline
operator==(const char* s1, SmallString const& t2)
    {
    return operator==(SmallString(s1),t2);
    }
bool inline
operator!=(SmallString const& t1, const char* s2)
    {
    return operator!=(t1,SmallString(s2));
    }
bool inline
operator!=(const char* s1, SmallString const& t2)
    {
    return operator!=(SmallString(s1),t2);
    }


void inline
write(std::ostream& s, SmallString const& t)
    {
    for(size_t n = 0; n < SmallString::size(); ++n)
        s.write((char*) &t[n],sizeof(char));
    }

void inline
read(std::istream& s, SmallString& t)
    {
    for(size_t n = 0; n < SmallString::size(); ++n)
        s.read((char*) &(t[n]),sizeof(char));
    }

inline SmallString::
SmallString()
    {
    name_.fill('\0');
    }

inline SmallString::
SmallString(const char* name)
    {
    name_.fill('\0');
    auto len = std::min(std::strlen(name),size());
#ifdef DEBUG
    if(std::strlen(name) > size())
        {
        std::cout << "Warning: SmallString name will be truncated to " << size() << " chars" << std::endl;
        }
#endif
    for(size_t j = 0; j < len; ++j)
        {
#ifdef DEBUG
        if(name[j]==',') throw std::runtime_error("SmallString cannot contain character ','");
#endif
        name_[j] = name[j];
        }
    assert(name_[size()]=='\0');
    }

} // namespace itensor

#undef CHECK_IND

#endif
