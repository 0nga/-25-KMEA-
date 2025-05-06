package nl.uu.trafficmas.agent.actions;

import java.util.List;
import java.util.Map;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.roadnetwork.AStar;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.Route;
import nl.uu.trafficmas.simulationmodel.AgentData;
import de.tudresden.sumo.cmd.Vehicle;
import de.tudresden.sumo.util.SumoCommand;
import de.tudresden.ws.container.SumoStringList;

public class ChangeRouteAction extends SumoAgentAction {

	public ChangeRouteAction(int priority) {
		super(priority);
		
	}

	@Override
	public boolean isRelevant(Agent agent) {
		boolean relevant = true;
		
		if(agent.getRoute().length == 1 || (agent.getDistance() < agent.getRoad().length*.8)){
			relevant = false;
		}
		
		
		return relevant;
	}

	@Override
	public double getTime(int currentTime, double currentSpeed,
			double meanTravelSpeedNextLane, double currentPos,
			double currentLaneLength, double maxComfySpeed,
			double routeRemainderLength, double leaderAgentSpeed,
			double leaderDistance, Agent agent) {
		Node nearestNode = agent.getRoute()[0].getToNode();
		Map<String,Double>roadBestLaneAverageTime = agent.getRoadNetwork().getRoadBestLaneAverageTravelTime();
		
		List<String> newRoute = AStar.findShortestPath(nearestNode, agent.getGoalNode(), agent.getRoadNetwork(), roadBestLaneAverageTime, agent.getMaxComfySpeed());
		
		// no possible route
		if(newRoute == null) {
			return Double.POSITIVE_INFINITY;
		}
		newRoute.add(0, agent.getRoute()[0].getID());
		agent.setPossibleNewRoute(newRoute);
		
		/*
		 * change route time 
		 * 
		 * lane_remainder       
		 * --------------   +   time on each road on the new route + current_time
		 * current_speed      
		 */
		
		double finishTime = currentTime;
		finishTime += (currentLaneLength-currentPos)/currentSpeed;
		Edge [] route = new Edge[newRoute.size()];
		int i = 0;
		for(String edgeID : newRoute) {
			route[i] = agent.getRoadNetwork().getEdge(edgeID);
			i++;
		}
		
		finishTime += Route.getRouteRemainderTime(route, agent.getRoadNetwork(), agent.getRoad(), agent.getMaxComfySpeed());
		
		return finishTime;
	}

	@Override
	public SumoCommand getCommand(Agent currentAgent) {
		currentAgent.setRoute(currentAgent.getPossibleNewRoute());
		currentAgent.setPossibleNewRoute(null);
		SumoStringList route = new SumoStringList();
		for(Edge edge : currentAgent.getRoute()){
			route.add(edge.getID());
		}
		return Vehicle.setRoute(currentAgent.agentID, route);
	}

	@Override
	public AgentData getNewAgentState(AgentData agentData) {
		// TODO make this better, look ahead?
		return agentData;
	}
}
