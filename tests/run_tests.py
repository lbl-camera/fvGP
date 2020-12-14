from test_fvgp import TestfvGP



def main():
    a = TestfvGP()
    ###choose a function for testing here
    a.test_1d_single_task(training_method = "global", N = 100)
    #a.test_1d_single_task(training_method = "mcmc", N = 100)
    #a.test_derivatives(0)
    #a.test_derivatives(1)
    #a.test_1d_single_task(training_method = "hgdl")
    #a.test_us_topo(method = "hgdl",dask_client = True)
    #a.test_us_topo(method = "global",dask_client = False)


if __name__ == "__main__":
    main()
