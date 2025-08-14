using TestToDoc

filepaths = [
    "test/cover.jl",
    "test/first_quant/spin_restricted_hartree_fock.jl",
    "test/periodic/dft.jl",
    "test/second_quant/fci.jl",
]

watch!(filepaths; src="./test", port=8001)
