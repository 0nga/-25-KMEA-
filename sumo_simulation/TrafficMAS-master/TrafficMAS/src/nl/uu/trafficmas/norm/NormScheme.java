package nl.uu.trafficmas.norm;

import java.util.List;
import java.util.Map;

import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public abstract class NormScheme {
	protected SanctionType sanctionType;
	public final List<Sensor> sensorList;
	public final String id;
	protected Map<String,String> attributes;
	public NormScheme(String id, SanctionType sanctionType, List<Sensor> sensorList) {
		this.id 			= id;
		this.sanctionType 	= sanctionType;
		this.sensorList		= sensorList;
	}
	
	public abstract List<NormInstantiation> instantiateNorms(RoadNetwork rn, int currentTime, Map<String, AgentData> currentOrgKnowledge);
	
	public abstract boolean checkCondition(Map<String, AgentData> currentOrgKnowledge);
	
	public abstract boolean violated(AgentData ad);

	public Sanction getSanction(String agentID) {
		return new Sanction(sanctionType, agentID);
	}

	public abstract boolean deadline(Map<String, AgentData> currentOrgKnowledge, int currentTime);

	public void addAttributes(Map<String, String> attributes) {
		this.attributes = attributes;
	}

	public abstract List<AgentData> getGoals();
}
