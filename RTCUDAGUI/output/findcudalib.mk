################################################################################
#
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
#  findgllib.mk is used to find the necessary GL Libraries for specific distributions
#               this is supported on Mac OSX and Linux Platforms
#
################################################################################

DISTRO  = $(shell lsb_release -i -s)

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

ifeq ($(ARMv7),1)
	OS_SIZE = 32
	OS_ARCH = armv7l
endif

ifeq ("$(OSUPPER)","LINUX")
    # $(info >>> findcudalib.mk -> LINUX path <<<)
    # Each set of Linux Distros have different paths (if using the Linux RPM/debian packages)
    ifeq ("$(DISTRO)","Ubuntu")
      ifeq ($(OS_SIZE),64)
        CUDAPATH  ?= /usr/lib/nvidia-current /usr/lib32/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current -L/usr/lib32/nvidia-current
      else
        CUDAPATH  ?= /usr/lib/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current
      endif
    endif
    ifeq ("$(DISTRO)","Debian")
      ifeq ($(OS_SIZE),64)
        CUDAPATH  ?= /usr/lib/nvidia-current /usr/lib32/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current -L/usr/lib32/nvidia-current
      else
        CUDAPATH  ?= /usr/lib/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current
      endif
    endif
    ifeq ("$(DISTRO)","SUSE")
      ifeq ($(OS_SIZE),64)
        CUDAPATH    ?= /usr/lib64 /usr/lib
        CUDALINK    ?= -L/usr/lib64 -L/usr/lib
		else
        CUDAPATH    ?= /usr/lib
        CUDALINK    ?= -L/usr/lib
      endif
    endif
    ifeq ("$(DISTRO)","Fedora")
      ifeq ($(OS_SIZE),64)
        CUDAPATH  ?= /usr/lib/nvidia-current /usr/lib32/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current -L/usr/lib32/nvidia-current
      else
        CUDAPATH  ?= /usr/lib/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current
      endif
    endif
    ifeq ("$(DISTRO)","Redhat")
      ifeq ($(OS_SIZE),64)
        CUDAPATH  ?= /usr/lib/nvidia-current /usr/lib32/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current -L/usr/lib32/nvidia-current
      else
        CUDAPATH  ?= /usr/lib/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current
      endif
    endif
    ifeq ("$(DISTRO)","CentOS")
      ifeq ($(OS_SIZE),64)
        CUDAPATH  ?= /usr/lib/nvidia-current /usr/lib32/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current -L/usr/lib32/nvidia-current
      else
        CUDAPATH  ?= /usr/lib/nvidia-current
        CUDALINK  ?= -L/usr/lib/nvidia-current
      endif
    endif
  
    ifeq ($(ARMv7),1)
      ifeq ($(OS_SIZE),64)
        CUDAPATH  += /usr/lib/x86_64-linux-gnu
        CUDALINK  += -L/usr/lib/x86_64-linux-gnu
      else
        CUDAPATH  += /usr/lib/i386-linux-gnu
        CUDALINK  += -L/usr/lib/i386-linux-gnu
      endif
    endif

  # find libcuda.so
  CUDALIB         ?= $(shell find $(CUDAPATH) -name libcuda.so  -print 2>/dev/null)

  ifeq ($(CUDALIB),'')
      $(info >>> WARNING - libcuda.so not found, CUDA Driver is not installed.  Please re-install the driver. <<<)
      EXEC=@echo "[@]"
  endif
else
  #$(info >>> findcudalib.mk -> DARWIN path <<<)
endif

