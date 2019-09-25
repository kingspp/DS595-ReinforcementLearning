from blackhc import mdp

spec = mdp.MDPSpec()
start = spec.state('start')
end = spec.state('end', terminal_state=True)
action_0 = spec.action()
action_1 = spec.action()

spec.transition(start, action_0, mdp.NextState(end))
spec.transition(start, action_1, mdp.NextState(end))
spec.transition(start, action_1, mdp.Reward(1))


spec_graph = spec.to_graph()
mdp.graph_to_png(spec_graph)
mdp.display_mdp(spec)


