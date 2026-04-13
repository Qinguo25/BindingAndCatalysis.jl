bin_PROGRAMS = \
scdd \
lcdd \
redcheck \
fourier \
projection \
adjacency \
allfaces \
cddexec

# Developer and regression-only utilities.
check_PROGRAMS = \
testcdd1 \
testshoot \
testcdd2 \
testlp1 \
testlp2 \
testlp3

# Build these binaries but do not install them; these are sample applications
# that are mostly interesting to developers.
noinst_PROGRAMS = \
redexter \
redundancies \
redundancies_clarkson

LDADD = ../lib-src/libcdd.la
AM_CPPFLAGS = -I$(top_srcdir)/lib-src

TESTS_ENVIRONMENT = top_srcdir='$(top_srcdir)' top_builddir='$(top_builddir)' srcdir='$(srcdir)'
TESTS = check-default.sh
EXTRA_DIST = check-default.sh check-logarithmic.sh check-logarithmic-large.sh

include ./Makefile.gmp.am

include ./Makefile.log.am
