from simulation.run import main


def test_main_for_linear_semi_parametric_sampling():
    args = ("--models LinearSemiParametricSampling --name test --steps 200 --save_every 2 "
          "--arms_nb 10 --a 0.5 --d 100 --reward_distribution normal --seed 1 "
            "--sigma_1 1 --sigma_2 1 --sigma_3 1").split(" ")
    main(args)


def test_main_for_beta_priors_sampling():
    args = ("--models BetaPriorsSampling --name test --steps 200 --save_every 2 "
            "--arms_nb 10 --a 0.5 --d 100 --reward_distribution binomial --seed 1").split(" ")
    main(args)


def test_main_for_gaussian_priors_sampling():
    args = ("--models GaussianPriorsSampling --name test --steps 200 --save_every 2 "
            "--arms_nb 10 --a 0.5 --d 100 --reward_distribution normal --seed 1").split(" ")
    main(args)


def test_main_for_linear_gaussian_sampling():
    args = ("--models LinearGaussianSampling --name test --steps 200 --save_every 2 "
            "--arms_nb 10 --a 0.5 --d 100 --reward_distribution normal --seed 1 --v 1").split(" ")
    main(args)


def test_main_for_all():
    args = ("--models LinearSemiParametricSampling LinearGaussianSampling GaussianPriorsSampling BetaPriorsSampling "
            "--name test_all --steps 200 --save_every 10 "
            "--arms_nb 10 --a 0.5 --d 100 --reward_distribution binomial --seed 1 --v 1 "
            "--sigma_1 1 --sigma_2 1 --sigma_3 1").split(" ")
    main(args)
