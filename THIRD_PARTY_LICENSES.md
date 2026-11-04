# Third-Party License Texts

Preserve this file with [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
when distributing the corresponding code. Component copyright statements and
special notices appear below; identical license terms are reproduced once in
this file or referenced from [LICENSE](LICENSE), with references from each
applicable component.

The dependency versions and selected DuckDB extensions are identified in the
notices file. Test and benchmark terms apply when that code is distributed.
Daemon and PerfWorks runtime dependencies, copyright attributions and license
terms are listed in [the daemon notices](https://github.com/NVIDIA/nsight-perf-daemon/blob/main/THIRD_PARTY_NOTICES.md).
Redistribute those notices along with any redistribution of the corresponding
components.

## Components

### FlatBuffers 23.5.26

Pinned upstream source archive. License: `Apache-2.0`.

```text
Copyright 2021 Google Inc. All rights reserved.
Copyright 2023 Google Inc. All rights reserved.
Copyright 2014 Google Inc. All rights reserved.
Copyright 2017 Google Inc. All rights reserved.
Copyright 2022 Google Inc. All rights reserved.
Copyright 2015 Google Inc. All rights reserved.
Copyright 2016 Google Inc. All rights reserved.
Copyright 2018 Dan Field
Copyright 2018 Google Inc. All rights reserved.
Copyright 2020 Google Inc. All rights reserved.
```

Applicable terms: [license text 1](#license-text-1).

### Protocol Buffers 3.21.12

Pinned upstream source archive. License: `BSD-3-Clause`.

```text
Copyright 2008 Google Inc.  All rights reserved.
Copyright 2015 Google Inc.  All rights reserved.
Copyright 2014 Google Inc.  All rights reserved.
Copyright (c) 2006, Google Inc.
All rights reserved.
Copyright 2012 Google Inc.  All rights reserved.
Copyright 2005 Google Inc.
All rights reserved.
Copyright 2009 Google Inc.  All rights reserved.
```

Applicable terms: [license text 2](#license-text-2).

### Perfetto schema baseline v58.3

Documented schema baseline 11fbaed836361258f23604daf67c837f3034e663. License: `Apache-2.0`.

```text
Copyright (c) 2017, The Android Open Source Project
Copyright 2015 The Chromium Authors
Copyright 2025 The syntaqlite Authors. All rights reserved.
Copyright 2002 Niels Provos <provos@citi.umich.edu>
All rights reserved.
Copyright (C) 2020 The Android Open Source Project
Copyright (C) 2018 The Android Open Source Project
Copyright (C) 2019 The Android Open Source Project
Copyright (C) 2017 The Android Open Source Project
```

Applicable terms: [license text 1](#license-text-1).

### DuckDB 1.5.5

Configured core, core_functions and parquet extensions. License: `MIT`.

```text
Copyright 2018-2025 Stichting DuckDB Foundation
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / Apache Arrow ADBC and nanoarrow

Embedded ADBC/nanoarrow source. License: `Apache-2.0`.

Applicable terms: [license text 1](#license-text-1).

```text
This product includes software developed at the Apache Software Foundation (https://www.apache.org/).
```

### DuckDB / robin-hood hashing

Adapted hashing code. License: `MIT`.

```text
Copyright (c) 2018-2021 Martin Ankerl
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / MD5 implementation

Colin Plumb implementation. License: `Public domain`.

```text
This code implements the MD5 message-digest algorithm.
The algorithm is due to Ron Rivest.  This code was
written by Colin Plumb in 1993, no copyright is claimed.
This code is in the public domain; do with it what you wish.

Equivalent code is available from RSA Data Security, Inc.
This code has been tested against that, and is equivalent,
except that you don't need to include two pages of legalese
with every copy.

To compute the message digest of a chunk of bytes, declare an
MD5Context structure, pass it to MD5Init, call MD5Update as
needed on buffers full of bytes, and then call MD5Final, which
will fill a supplied 16-byte array with the digest.
```

### DuckDB / Salesforce merge-sort tree

Adapted source in execution/merge_sort_tree.hpp. License: `MIT`.

```text
Copyright 2022 salesforce.com, inc.
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / ALP and ALP-RD

Embedded compression algorithms. License: `MIT`.

```text
Copyright (c) 2023 CWI, Azim Afroozeh,  Leonardo Xavier Kuffo Rivero
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / brotli

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors.
Copyright 2016 Google Inc. All Rights Reserved.
Copyright 2013 Google Inc. All Rights Reserved.
Copyright 2017 Google Inc. All Rights Reserved.
Copyright 2015 Google Inc. All Rights Reserved.
Copyright 2014 Google Inc. All Rights Reserved.
Copyright 2010 Google Inc. All Rights Reserved.
Copyright 2018 Google Inc. All Rights Reserved.
Copyright 2022 Google Inc. All Rights Reserved.
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / concurrentqueue

Bundled with DuckDB 1.5.5. License: `BSD-2-Clause; zlib for the semaphore`.

```text
Copyright (c) 2013-2016, Cameron Desrochers.
All rights reserved.
Copyright (c) 2015 Jeff Preshing
```

Applicable terms: [license text 4](#license-text-4), [license text 5](#license-text-5).

### DuckDB / fast_float

Bundled with DuckDB 1.5.5. License: `MIT`.

Applicable terms: [license text 3](#license-text-3).

```text
duckdb_fast_float by Daniel Lemire
duckdb_fast_float by João Paulo Magalhaes


with contributions from Eugene Golushkov
with contributions from Maksim Kita
with contributions from Marcin Wojdyr
with contributions from Neal Richardson
with contributions from Tim Paine
with contributions from Fabio Pellacini
```

### DuckDB / fastpforlib

Bundled with DuckDB 1.5.5. License: `Apache-2.0`.

Applicable terms: [license text 1](#license-text-1).

### DuckDB / fmt

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2012 - present, Victor Zverovich
Copyright (c) 2012 - 2016, Victor Zverovich
All rights reserved.
Copyright (c) 2012 - present, Victor Zverovich
All rights reserved.
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / fsst

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2018-2020, CWI, TU Munich, FSU Jena
the API for FSST compression -- (c) Peter Boncz, Viktor Leis and Thomas Neumann (CWI, TU Munich), 2018-2019
Copyright 2018-2020, CWI, TU Munich, FSU Jena
```

Applicable terms: [license text 3](#license-text-3), [license text 6](#license-text-6).

```text
the API for FSST compression -- (c) Peter Boncz, Viktor Leis and Thomas Neumann (CWI, TU Munich), 2018-2019
```

### DuckDB / httplib

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2017 yhirose
Copyright (c) 2025 Yuji Hirose. All rights reserved.
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / hyperloglog

Bundled with DuckDB 1.5.5. License: `MIT; BSD-3-Clause for Redis-derived code`.

```text
Copyright (c) 2012 Art.sy, Inc.
Copyright (c) 2014, Salvatore Sanfilippo <antirez at gmail dot com>
All rights reserved.
Copyright (c) 2006-2015, Salvatore Sanfilippo <antirez at gmail dot com>
Copyright (c) 2015, Oran Agra
Copyright (c) 2015, Redis Labs, Inc
All rights reserved.
```

Applicable terms: [license text 3](#license-text-3), [license text 7](#license-text-7).

### DuckDB / jaro_winkler

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright © 2022 Max Bachmann
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / libpg_query

Bundled with DuckDB 1.5.5. License: `BSD-style; Bison skeleton exception`.

```text
Copyright (c) 2015, Lukas Fittl <lukas@fittl.com>
All rights reserved.
Portions Copyright (c) 1996-2017, PostgreSQL Global Development PGGroup
Portions Copyright (c) 1994, Regents of the University of California
Copyright (c) 2003-2017, PostgreSQL Global Development PGGroup
Copyright (c) 2014-2017, PostgreSQL Global Development PGGroup
Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
Free Software Foundation, Inc.
Portions Copyright (c) 1996-2015, PostgreSQL Global Development PGGroup
Portions Copyright (c) 1996-2018, PostgreSQL Global Development Group
```

Applicable terms: [license text 8](#license-text-8), [license text 9](#license-text-9), [license text 10](#license-text-10), [license text 11](#license-text-11).

### DuckDB / lz4

Bundled with DuckDB 1.5.5. License: `BSD-2-Clause`.

```text
Copyright (c) 2011-2020, Yann Collet
All rights reserved.
Copyright (C) 2011-2020, Yann Collet.
```

Applicable terms: [license text 12](#license-text-12), [license text 13](#license-text-13).

### DuckDB / mbedtls

Bundled with DuckDB 1.5.5. License: `Apache-2.0`.

```text
Copyright The Mbed TLS Contributors
```

Applicable terms: [license text 1](#license-text-1).

### DuckDB / miniz

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright 2013-2014 RAD Game Tools and Valve Software
Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC
Copyright 2016 Martin Raiber
All Rights Reserved.
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / parquet

Bundled with DuckDB 1.5.5. License: `Apache-2.0`.

Applicable terms: [license text 1](#license-text-1).

```text
This product includes code from Apache Avro.

Copyright: 2014 The Apache Software Foundation.
Home page: https://avro.apache.org/
License: http://www.apache.org/licenses/LICENSE-2.0
```

```text
This project includes code from Daniel Lemire's JavaFastPFOR project. The
"Lemire" bit packing source code produced by parquet-generator is derived from
the JavaFastPFOR project.

Copyright: 2013 Daniel Lemire
Home page: http://lemire.me/en/
Project page: https://github.com/lemire/JavaFastPFOR
License: Apache License Version 2.0 http://www.apache.org/licenses/LICENSE-2.0
```

### DuckDB / pcg

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2014-2017 Melissa O'Neill and PCG Project contributors
Copyright 2014-2017 Melissa O'Neill <oneill@pcg-random.org>,
and the PCG Project contributors.
Copyright 2014-2019 Melissa O'Neill <oneill@pcg-random.org>,
and the PCG Project contributors.
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / pdqsort

Bundled with DuckDB 1.5.5. License: `zlib`.

```text
Copyright (c) 2021 Orson Peters <orsonpeters@gmail.com>
Copyright (c) 2021 Orson Peters
```

Applicable terms: [license text 14](#license-text-14).

### DuckDB / re2

Bundled with DuckDB 1.5.5. License: `BSD-3-Clause; Lucent permission notice`.

```text
Copyright (c) 2009 The RE2 Authors. All rights reserved.
Copyright 2023 The RE2 Authors.  All Rights Reserved.
Copyright 2016 The RE2 Authors.  All Rights Reserved.
Copyright 2008 The RE2 Authors.  All Rights Reserved.
Copyright 2007 The RE2 Authors.  All Rights Reserved.
Copyright 2009 The RE2 Authors.  All Rights Reserved.
Copyright 2006-2007 The RE2 Authors.  All Rights Reserved.
Copyright 2006 The RE2 Authors.  All Rights Reserved.
Copyright 2018 The RE2 Authors.  All Rights Reserved.
Copyright 2003-2009 The RE2 Authors.  All Rights Reserved.
Copyright 2010 The RE2 Authors.  All Rights Reserved.
Copyright 2004 The RE2 Authors.  All Rights Reserved.
Copyright 2001-2010 The RE2 Authors.  All Rights Reserved.
Copyright (c) 2002 by Lucent Technologies.
Copyright 1999-2005 The RE2 Authors.  All Rights Reserved.
```

Applicable terms: [license text 2](#license-text-2), [license text 15](#license-text-15).

### DuckDB / ska_sort

Bundled with DuckDB 1.5.5. License: `BSL-1.0`.

```text
Copyright Malte Skarupke 2016.
```

Applicable terms: [license text 16](#license-text-16).

### DuckDB / skiplist

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2015-2023 Paul Ross. All rights reserved.
Copyright (c) 2017-2023 Paul Ross
Copyright (c) 2015-2023 Paul Ross
Copyright (c) 2017 Paul Ross. All rights reserved.
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / snappy

Bundled with DuckDB 1.5.5. License: `BSD-3-Clause`.

```text
Copyright 2011, Google Inc.
All rights reserved.
Copyright 2008 Google Inc. All Rights Reserved.
Copyright 2011 Google Inc. All Rights Reserved.
Copyright 2005 Google Inc. All Rights Reserved.
Copyright 2005 and onwards Google Inc.
```

Applicable terms: [license text 2](#license-text-2).

### DuckDB / tdigest

Bundled with DuckDB 1.5.5. License: `Apache-2.0`.

Applicable terms: [license text 1](#license-text-1).

```text
The Java version of the t-digest was originally authored by Ted Dunning

A number of small but very helpful changes have been contributed by Adrien Grand (https://github.com/jpountz) to the Java version.

The C++ version herein is a derivative of the Java version.  It was written by Derrick R. Burns (https:://github.com/derrickburns).
The main modifications are 1) higher performance multi- t-digest merging and 2) faster quantile() and cdf() computation.
```

### DuckDB / thrift

Bundled with DuckDB 1.5.5. License: `Apache-2.0`.

Applicable terms: [license text 1](#license-text-1).

### DuckDB / utf8proc

Bundled with DuckDB 1.5.5. License: `MIT; Unicode data terms`.

```text
Copyright &copy; 2014-2019 by Steven G. Johnson, Jiahao Chen, Tony Kelman, Jonas Fonseca, and other contributors listed in the git history.
Copyright (c) 2009, 2013 Public Software Group e. V., Berlin, Germany
Copyright (c) 1991-2007 Unicode, Inc. All rights reserved. Distributed
under the Terms of Use in http://www.unicode.org/copyright.html.*
Copyright (c) 2014-2021 Steven G. Johnson, Jiahao Chen, Peter Colberg, Tony Kelman, Scott P. Jones, and other contributors.
Copyright (c) 2009 Public Software Group e. V., Berlin, Germany
```

Applicable terms: [license text 17](#license-text-17), [license text 3](#license-text-3).

### DuckDB / vergesort

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2015 Morwenn <morwenn29@hotmail.fr>
Copyright (c) 2015 Orson Peters
Copyright (c) 2015-2016 Morwenn <morwenn29@hotmail.fr>
Copyright (c) 2015-2020 Morwenn <morwenn29@hotmail.fr>
```

Applicable terms: [license text 3](#license-text-3), [license text 14](#license-text-14).

### DuckDB / yyjson

Bundled with DuckDB 1.5.5. License: `MIT`.

```text
Copyright (c) 2020 YaoYuan <ibireme@gmail.com>
```

Applicable terms: [license text 3](#license-text-3).

### DuckDB / zstd

Bundled with DuckDB 1.5.5. License: `BSD-3-Clause; MIT for divsufsort`.

```text
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
Copyright (c) Meta Platforms, Inc. and affiliates.
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
Copyright (c) 2016 Tino Reichardt
All rights reserved.
Copyright (c) 2012-2020, Yann Collet, Facebook, Inc.
Copyright (c) 2003-2008 Yuta Mori All Rights Reserved.
```

Applicable terms: [license text 18](#license-text-18), [license text 3](#license-text-3).

## License texts

### License text 1

Apache-2.0: see the full text in [LICENSE](LICENSE).

The same license text applies to the Apache-2.0 components identified above.
Preserve LICENSE with this file when redistributing those components.

### License text 2

BSD-3-Clause.

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### License text 3

MIT.

```text
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### License text 4

BSD-2-Clause; zlib for the semaphore.

```text
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### License text 5

BSD-2-Clause; zlib for the semaphore.

```text
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgement in the product documentation would be
	appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

### License text 6

MIT.

```text
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### License text 7

MIT; BSD-3-Clause for Redis-derived code.

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
Neither the name of Redis nor the names of its contributors may be used
    to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### License text 8

BSD-style; Bison skeleton exception.

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of pg_query nor the names of its contributors may be used
to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### License text 9

BSD-style; Bison skeleton exception.

```text
Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.
```

### License text 10

BSD-style; Bison skeleton exception.

```text
As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.
```

### License text 11

BSD-style; Bison skeleton exception.

```text
Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.
```

### License text 12

BSD-2-Clause.

```text
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### License text 13

BSD-2-Clause.

```text
Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### License text 14

zlib.

```text
Permission is granted to anyone to use this software for any purpose, including commercial
applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the
   original software. If you use this software in a product, an acknowledgment in the product
   documentation would be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be misrepresented as
   being the original software.

3. This notice may not be removed or altered from any source distribution.
```

### License text 15

BSD-3-Clause; Lucent permission notice.

```text
The authors of this software are Rob Pike and Ken Thompson.
             Copyright (c) 2002 by Lucent Technologies.
Permission to use, copy, modify, and distribute this software for any
purpose without fee is hereby granted, provided that this entire notice
is included in all copies of any software which is or includes a copy
or modification of this software and in all copies of the supporting
documentation for such software.
THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
```

### License text 16

BSL-1.0.

```text
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

### License text 17

MIT; Unicode data terms.

```text
## utf8proc license ##

**utf8proc** is a software package originally developed
by Jan Behrens and the rest of the Public Software Group, who
deserve nearly all of the credit for this library, that is now maintained by the Julia-language developers.  Like the original utf8proc,
whose copyright and license statements are reproduced below, all new
work on the utf8proc library is licensed under the [MIT "expat"
license](http://opensource.org/licenses/MIT):

*Copyright &copy; 2014-2019 by Steven G. Johnson, Jiahao Chen, Tony Kelman, Jonas Fonseca, and other contributors listed in the git history.*

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## Original utf8proc license ##

*Copyright (c) 2009, 2013 Public Software Group e. V., Berlin, Germany*

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## Unicode data license ##

This software contains data (`utf8proc_data.c`) derived from processing
the Unicode data files. The following license applies to that data:

**COPYRIGHT AND PERMISSION NOTICE**

*Copyright (c) 1991-2007 Unicode, Inc. All rights reserved. Distributed
under the Terms of Use in http://www.unicode.org/copyright.html.*

Permission is hereby granted, free of charge, to any person obtaining a
copy of the Unicode data files and any associated documentation (the "Data
Files") or Unicode software and any associated documentation (the
"Software") to deal in the Data Files or Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, and/or sell copies of the Data Files or Software, and
to permit persons to whom the Data Files or Software are furnished to do
so, provided that (a) the above copyright notice(s) and this permission
notice appear with all copies of the Data Files or Software, (b) both the
above copyright notice(s) and this permission notice appear in associated
documentation, and (c) there is clear notice in each modified Data File or
in the Software as well as in the documentation associated with the Data
File(s) or Software that the data or software has been modified.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS
INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR
CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THE DATA FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

Unicode and the Unicode logo are trademarks of Unicode, Inc., and may be
registered in some jurisdictions. All other trademarks and registered
trademarks mentioned herein are the property of their respective owners.
```

### License text 18

BSD-3-Clause; MIT for divsufsort.

```text
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

Neither the name Facebook nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

