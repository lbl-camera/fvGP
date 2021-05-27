from test_fvgp import TestfvGP



def main():
    a = TestfvGP()
    a.test_initialization()
    ###choose a function for testing here
    a.test_1d_single_task(training_method = "global", N = 100)
    #a.test_1d_multi_task(training_method = "global", N = 100)
    #a.test_1d_single_task(training_method = "mcmc", N = 100)
    #a.test_1d_single_task_async()
    #a.test_us_topo(method = "hgdl")


if __name__ == "__main__":
    main()
