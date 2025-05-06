package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class MaxSpeedNormScheme extends NormScheme {


	private double maxSpeed;
	private List<AgentData> goals;
	public Set<String> agentNorms;
	private String instantiateAtLane;
	public String deadlineAtRoad;
	public MaxSpeedNormScheme(String id, SanctionType sanctionType,
			List<Sensor> sensorList) {
		super(id, sanctionType, sensorList);
		maxSpeed = 80/3.6;
		goals = new ArrayList<AgentData>();
		agentNorms = new HashSet<String>();
		// TODO Auto-generated constructor stub
	}

	@Override
	public List<NormInstantiation> instantiateNorms(RoadNetwork rn, int currentTime,
			Map<String, AgentData> currentOrgKnowledge) {
		
		List<NormInstantiation> list = new ArrayList<NormInstantiation>();

		for(AgentData agentData : currentOrgKnowledge.values()) {
			Lane l = rn.getEdge(agentData.roadID).getRoad().getLanes()[agentData.laneIndex];
			if(!agentNorms.contains(agentData.id) && l.getID().equals(instantiateAtLane)) {
				list.add(new MaxSpeedNormInstantiation(this, agentData.id));
				agentNorms.add(agentData.id);
			}
		}
		
		return list;
	}

	
	@Override
	public boolean checkCondition(Map<String, AgentData> currentOrgKnowledge) {
		
		return true;
	}

	@Override
	public boolean violated(AgentData ad) {
		// TODO Auto-generated method stub
		return ad.velocity > maxSpeed;
	}


	@Override
	public boolean deadline(Map<String, AgentData> currentOrgKnowledge,
			int currentTime) {
		
		return false;
	}
	
	@Override
	public void addAttributes(Map<String, String> attributes) {
		this.attributes = attributes;
		if(attributes.containsKey("maxspeed")) {
			maxSpeed = Double.parseDouble(attributes.get("maxspeed"));
			goals.add(new AgentData(null, null, -1, maxSpeed, null, 1, -1, -1));
		}
		
		if(attributes.containsKey("instantiateAtLane")) {
			instantiateAtLane = attributes.get("instantiateAtLane");
		}
		
		if(attributes.containsKey("deadlineAtRoad")) {
			deadlineAtRoad = attributes.get("deadlineAtRoad");
		}
	}


	@Override
	public List<AgentData> getGoals() {
		return goals;
	}
	
}
