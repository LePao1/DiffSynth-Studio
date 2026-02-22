def VaceWanModelDictConverter(state_dict):
    return {name: state_dict[name] for name in state_dict if name.startswith("vace")}
