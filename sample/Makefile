include ../this_dir.mk
include ../options.mk

#Define Flags ----------

TENSOR_HEADERS=$(PREFIX)/itensor/all.h
CCFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(CPPFLAGS) $(OPTIMIZATIONS)
CCGFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS)
LIBFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBFLAGS)
LIBGFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBGFLAGS)

#Rules ------------------

%.o: %.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: dmrg dmrg_table dmrgj1j2 exthubbard trg mixedspin hubbard_2d tree_dmrg testmps bosehubbard ising 1dflashratchet

debug: dmrg-g dmrg_table-g dmrgj1j2-g exthubbard-g trg-g mixedspin-g hubbard_2d-g tree_dmrg-g testmps-g bosehubbard-g ising-g 1dflashratchet-g

all: dmrg dmrg_table dmrgj1j2 exthubbard trg mixedspin hubbard_2d tree_dmrg testmps bosehubbard ising 1dflashratchet

dmrg: dmrg.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) dmrg.o -o dmrg $(LIBFLAGS)

dmrg-g: mkdebugdir .debug_objs/dmrg.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/dmrg.o -o dmrg-g $(LIBGFLAGS)

mixedspin: mixedspin.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) mixedspin.o -o mixedspin $(LIBFLAGS)

mixedspin-g: mkdebugdir .debug_objs/mixedspin.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/mixedspin.o -o mixedspin-g $(LIBGFLAGS)

dmrg_table: dmrg_table.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) dmrg_table.o -o dmrg_table $(LIBFLAGS)

dmrg_table-g: mkdebugdir .debug_objs/dmrg_table.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/dmrg_table.o -o dmrg_table-g $(LIBGFLAGS)

dmrgj1j2: dmrgj1j2.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) dmrgj1j2.o -o dmrgj1j2 $(LIBFLAGS)

dmrgj1j2-g: mkdebugdir .debug_objs/dmrgj1j2.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) .debug_objs/dmrgj1j2.o -o dmrgj1j2-g $(LIBGFLAGS)

exthubbard: exthubbard.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) exthubbard.o -o exthubbard $(LIBFLAGS)

exthubbard-g: mkdebugdir .debug_objs/exthubbard.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) .debug_objs/exthubbard.o -o exthubbard-g $(LIBGFLAGS)

trg: trg.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) trg.o -o trg $(LIBFLAGS)

trg-g: mkdebugdir .debug_objs/trg.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/trg.o -o trg-g $(LIBGFLAGS)

hubbard_2d: hubbard_2d.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) hubbard_2d.o -o hubbard_2d $(LIBFLAGS)

hubbard_2d-g: mkdebugdir .debug_objs/hubbard_2d.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/hubbard_2d.o -o hubbard_2d-g $(LIBGFLAGS)

tree_dmrg: tree_dmrg.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) tree_dmrg.o -o tree_dmrg $(LIBFLAGS)

tree_dmrg-g: mkdebugdir .debug_objs/tree_dmrg.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/tree_dmrg.o -o tree_dmrg-g $(LIBGFLAGS)

testmps: testmps.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) testmps.o -o testmps $(LIBFLAGS)

testmps-g: mkdebugdir .debug_objs/testmps.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/testmps.o -o testmps-g $(LIBGFLAGS)

bosehubbard: bosehubbard.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) bosehubbard.o -o bosehubbard $(LIBFLAGS)

bosehubbard-g: mkdebugdir .debug_objs/bosehubbard.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/bosehubbard.o -o bosehubbard-g $(LIBGFLAGS)

ising: ising.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) ising.o -o ising $(LIBFLAGS)

ising-g: mkdebugdir .debug_objs/ising.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/ising.o -o ising-g $(LIBGFLAGS)

1dflashratchet: 1dflashratchet.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) 1dflashratchet.o -o 1dflashratchet $(LIBFLAGS)

1dflashratchet-g: mkdebugdir .debug_objs/1dflashratchet.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/1dflashratchet.o -o 1dflashratchet-g $(LIBGFLAGS)

1dflashratchet_serial: 1dflashratchet_serial.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) 1dflashratchet_serial.o -o 1dflashratchet_serial $(LIBFLAGS)

1dflashratchet_serial-g: mkdebugdir .debug_objs/1dflashratchet_serial.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/1dflashratchet_serial.o -o 1dflashratchet_serial-g $(LIBGFLAGS)

mkdebugdir:
	mkdir -p .debug_objs

clean:
	@rm -fr *.o .debug_objs dmrg dmrg-g \
	dmrg_table dmrg_table-g dmrgj1j2 dmrgj1j2-g exthubbard exthubbard-g \
    mixedspin mixedspin-g trg trg-g hubbard_2d hubbard_2d-g tree_dmrg tree_dmrg-g \
    testmp testmp-g bosehubbard bosehubbard-g ising ising-g \
    1dflashratchet 1dflashratchet-g 1dflashratchet_serial 1dflashratchet_serial-g
