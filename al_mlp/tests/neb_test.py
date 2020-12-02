# https://wiki.fysik.dtu.dk/ase/tutorials/neb/diffusion.html#diffusion-tutorial
# Surface Diffusion Energy Barriers
# Building your structure


def construct_geometries(parent_calc, ml2relax):
    counter_calc = CounterCalc(parent_calc, "parent_database")
    # Initial structure guess
    initial_slab = fcc100("Cu", size=(2, 2, 3))
    add_adsorbate(initial_slab, "O", 1.7, "hollow")
    initial_slab.center(axis=2, vacuum=4.0)
    mask = [atom.tag > 1 for atom in initial_slab]
    initial_slab.set_constraint(FixAtoms(mask=mask))

    initial_slab.set_calculator(counter_calc)

    # Final structure guess
    final_slab = initial_slab.copy()
    final_slab[-1].x += final_slab.get_cell()[0, 0] / 3
    final_slab.set_calculator(counter_calc)

    if not ml2relax:
        print("BUILDING INITIAL")
        qn = BFGS(
            initial_slab, trajectory="initial.traj", logfile="initial_relax_log.txt"
        )
        qn.run(fmax=0.01, steps=100)
        print("BUILDING FINAL")
        qn = BFGS(final_slab, trajectory="final.traj", logfile="final_relax_log.txt")
        qn.run(fmax=0.01, steps=100)
        initial_slab = read("initial.traj", "-1")
        final_slab = read("final.traj", "-1")
        # If there is already a pre-existing initial and final relaxed parent state we can read that to use as a starting point
        # initial_slab = read("/content/parent_initial.traj")
        # final_slab = read("/content/parent_final.traj")
    else:
        initial_slab = attach_sp_calc(initial_slab)
        final_slab = attach_sp_calc(final_slab)

    initial_force_calls = counter_calc.force_calls
    return initial_slab, final_slab, initial_force_calls
