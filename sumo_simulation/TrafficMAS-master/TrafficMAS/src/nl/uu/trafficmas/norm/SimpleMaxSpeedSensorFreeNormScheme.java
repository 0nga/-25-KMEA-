package nl.uu.trafficmas.norm;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import nl.uu.trafficmas.roadnetwork.Lane;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Sensor;
import nl.uu.trafficmas.simulationmodel.AgentData;

public class SimpleMaxSpeedSensorFreeNormScheme extends NormScheme {


	private double maxSpeed;
	private List<AgentData> goals;
	public Set<String> agentNorms;
	private String firstLaneId;
	private String secondLaneId;
	private String thirdLaneId;
	private String deadlineRoad;
	private int maxPeopleOnLane;
	public SimpleMaxSpeedSensorFreeNormScheme(String id, SanctionType sanctionType,
			List<Sensor> sensorList) {
		super(id, sanctionType, sensorList);
		maxSpeed = 4;
		goals = new ArrayList<AgentData>();
		agentNorms = new HashSet<String>();
		maxPeopleOnLane = 2;
	}


	@Override
	public List<NormInstantiation> instantiateNorms(RoadNetwork rn, int currentTime,
			Map<String, AgentData> currentOrgKnowledge) {
		
		if(goals.isEmpty()) {
			Road r = rn.getRoadFromID(deadlineRoad);
			Lane l = null;
			
			// find lane agent needs to move to
			for(Lane l2 : r.getLanes()) {
				if(l2.getID().equals(thirdLaneId)) {
					l = l2;
					break;
				}
			}
					
			int position = 1;
			double speed = 40/3.6;
			int laneIndex = l.laneIndex;
			int deceleration = -1;
			int acceleration = -1;
			goals.add(new AgentData(null, null, position, speed, null, 0, deceleration, acceleration));
			goals.add(new AgentData(null, null, position, -1, null, laneIndex, deceleration, acceleration));
		}
		
		
		List<NormInstantiation> list = new ArrayList<NormInstantiation>();
		
		// count number of agents on first lane
		int agentsFirstLaneCount = 0;
		for (Entry<String, AgentData> entry : currentOrgKnowledge.entrySet()) {
			AgentData agentData = entry.getValue();
			Lane l = rn.getRoadFromID(agentData.roadID).getLanes()[agentData.laneIndex];
			boolean isOnFirstLane = l.getID().equals(firstLaneId);
			if(isOnFirstLane) {
				agentsFirstLaneCount++;
			}
		}
		
		// TODO: 2 is arbitrary here
		
		if(agentsFirstLaneCount > maxPeopleOnLane) {
			// add norm to agents that are on the first lane
			for (Entry<String, AgentData> entry : currentOrgKnowledge.entrySet()) {
				AgentData agentData = entry.getValue();
				Lane l = rn.getRoadFromID(agentData.roadID).getLanes()[agentData.laneIndex];
				boolean isOnFirstLane = l.getID().equals(firstLaneId);
				if(isOnFirstLane) {
					NormInstantiation ni = new SimpleMaxSpeedSensorFreeNormInstantiation(this, agentData.id);
					ni.addRoadNetwork(rn);
					list.add(ni);
					agentNorms.add(agentData.id);
				}
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
				
		firstLaneId = attributes.get("firstlane");
		secondLaneId =  attributes.get("secondlane");
		thirdLaneId =  attributes.get("thirdlane");
		deadlineRoad = attributes.get("deadlineroad");
		
		if(attributes.get("maxPeopleOnLane")!= null) {
			maxPeopleOnLane = Integer.parseInt(attributes.get("maxPeopleOnLane"));		
		}

	}


	@Override
	public List<AgentData> getGoals() {
		return goals;
	}


	public String getDeadlineRoad() {
		return deadlineRoad;
	}



	public String getSecondLaneId() {
		return secondLaneId;
	}
	
}
