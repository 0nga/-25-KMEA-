package nl.uu.trafficmas.agent.actions;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Route;
import nl.uu.trafficmas.simulationmodel.AgentData;
import de.tudresden.sumo.cmd.Vehicle;
import de.tudresden.sumo.util.SumoCommand;

public class ChangeRightLaneAction extends SumoAgentAction {
	public static final int OVERTAKE_DURATION = 5;

	public ChangeRightLaneAction(int priority) {
		super(priority);
	}

	@Override
	public double getTime(int currentTime, double currentSpeed,
			double meanTravelSpeedNextLane, double currentPos,
			double currentLaneLength, double maxComfySpeed,
			double routeRemainderLength, double leaderAgentSpeed,
			double leaderDistance, Agent agent) {
		/*
		 * change lane time (no knowledge about RN)
		 * 
		 * lane_remainder       
		 * --------------   +   time on each road on the remaining route + current_time
		 * next_lane_speed      
		 */
		// change lane time:
	
		double finishTime = currentTime;
		double meanOrComfySpeedNextLane = Math.min(meanTravelSpeedNextLane,maxComfySpeed);
		double laneDistRemaining = (currentLaneLength-currentPos);
		double timeSpentOnNextLane = laneDistRemaining/meanOrComfySpeedNextLane;
		finishTime += timeSpentOnNextLane;
		
		
		Edge[] route = agent.getRoute();
		finishTime += Route.getRouteRemainderTime(route, agent.getRoadNetwork(), agent.getRoad(), agent.getMaxComfySpeed());
		
		return finishTime;
	}


	@Override
	public boolean isRelevant(Agent agent) {		
		return agent.getLane().hasRightLane();
	}

	@Override
	public SumoCommand getCommand(Agent currentAgent) {
		byte agentLaneIndex = currentAgent.getLane().laneIndex;
		int maxLaneIndex = currentAgent.getRoad().laneList.size()-1;
		// cannot change
		if( agentLaneIndex >= maxLaneIndex) {
			return null;
		}
			
		return Vehicle.changeLane(currentAgent.agentID, (byte) (agentLaneIndex+1) , OVERTAKE_DURATION);
	}

	@Override
	public AgentData getNewAgentState(AgentData agentData) {
		// TODO take into account that there is no next lane
		Object [] leader = {agentData.leaderId, agentData.leaderDistance};
		AgentData newData = new AgentData(agentData.id, leader, agentData.position, (agentData.velocity), agentData.roadID, agentData.laneIndex+1,Agent.deceleration, Agent.acceleration);	
		
		return newData;
	}

}
