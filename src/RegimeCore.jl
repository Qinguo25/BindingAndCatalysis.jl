get_binding_network(model::CatalysisData) = model.bn
get_binding_network(model::Bnc,args...)=model



get_binding_network(rgm::CatalysisRegime) = get_binding_network(rgm.network)
get_binding_network(rgm::BindRegime,args...)=get_binding_network(rgm.network)
get_binding_network(rgm::BncRegime,args...)=get_binding_network(rgm.bind_rgm)

get_catalysis_network(model::CatalysisData) = model.cn