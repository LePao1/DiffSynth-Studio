def WanAnimateAdapterStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith(("pose_patch_embedding.", "face_adapter", "face_encoder", "motion_encoder")):
            state_dict_[name] = state_dict[name]
    return state_dict_
