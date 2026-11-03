"""Run point-foot policy validation in native CPU MuJoCo."""

if __package__:
    from .pointleg_sim2sim import main
else:
    from pointleg_sim2sim import main


if __name__ == "__main__":
    main()
