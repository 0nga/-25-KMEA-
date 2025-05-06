package nl.uu.trafficmas.simulationmodel;

import java.util.HashMap;
import java.util.Random;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.actions.AgentAction;
import nl.uu.trafficmas.datamodel.MASData;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

public interface SimulationModel {
	/**
	 * Initializes the connection to SUMO with default settings.
	 */
	public void initialize(); 
	
	/**
	 * Initializes the connection to SUMO with settings as provided in 'optionValueMap'
	 * @param optionValueMap
	 */
	public void initializeWithOptions(HashMap<String, String> optionValueMap);
	
	/**
	 * Closes the connection to the SUMO application.
	 */
	public void close();
	
	/**
	 * Does a timestep in the SUMO application.
	 */
	public void doTimeStep();
	
	/**
	 * Does a timestep in SUMO and returns StateData, which contains information about Edge, Lane and Agent objects.
	 * @return a StateData object which contains 3 HashMaps concerning agent, edge and lane data, and an Integer with the current timestep.
	 */
	public StateData getNewStateData();
	
	public RoadNetwork updateRoadNetworkLanes(RoadNetwork rn);
	
	/**
	 * Sends a list of commands to spawn Agents to SUMO. The agent data and spawn times are provided in 'agentPairList'. 
	 * Default values depending on the agentProfileType are added, like Color and MaxSpeed.
	 * @param agentPairList
	 * @return the HashMap containing all agents that will spawn in the simulation. 
	 */
	public HashMap<String, Agent> addAgents(HashMap<Agent, Integer> agentPairList, Random rng, double rightLaneRatio);
	
	/**
	 * Adds a single agent 'agent' at time 'tick' with route 'routeID'.
	 * @param agent
	 * @param routeID
	 * @param tick
	 */
	public void addAgent(Agent agent, String routeID, int tick);

	/**
	 * Sends a list of commands to SUMO. The commands are generated from the HashMap 'agentActions' and are all actions the vehicle can take.
	 * The following actions can be send to SUMO:
	 * ChangeLane,
	 * ChangeRoad,
	 * ChangeVelocity5,
	 * ChangeVelocity10,
	 * ChangeVelocity20.
	 * @param actions
	 * @param conn
	 * @throws Exception 
	 */
	public void simulateAgentActions(HashMap<Agent, AgentAction> agentActions) throws Exception;
	public HashMap<String, Agent> updateCurrentAgentMap(HashMap<String, Agent> completeAgentMap, HashMap<String, Agent> oldAgentMap);
}
