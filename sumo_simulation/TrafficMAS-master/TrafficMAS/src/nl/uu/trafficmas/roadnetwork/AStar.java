package nl.uu.trafficmas.roadnetwork;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class AStar {
	public static List<String> findShortestPath(Node startNode, Node finalNode, RoadNetwork roadNetwork, Map<String, Double> averageTravelTime, double maxSpeed){
		if(startNode.equals(finalNode)) {
			return null;
		}
		
		Set<Node> openList = new HashSet<Node>();
		Set<Node> closedList = new HashSet<Node>();
		Map<Node,Double> f = new HashMap<Node,Double>();
		Map<Node,Double> g = new HashMap<Node,Double>();
		Map<Node,Node> parents = new HashMap<Node, Node>();
		
		g.put(startNode, 0.0);
		f.put(startNode, g.get(startNode) + AStar.heuristic(startNode, finalNode, maxSpeed));
		openList.add(startNode);
		Node currentNode = null;
		
		while(!openList.isEmpty()) {
			currentNode = null;
			// get the node with the lowest f-score
			for(Node node : openList) {
				if(currentNode == null || f.get(node) < f.get(currentNode)) {
					currentNode = node;
				}
			}
		
			if(currentNode.equals(finalNode)) {
				return reconstructPath(parents,finalNode, roadNetwork);
			}
			
			openList.remove(currentNode);
			closedList.add(currentNode);
			
			Set<Node> nodes = roadNetwork.getSuccessors(currentNode);
			for(Node successor : nodes) {
				if(closedList.contains(successor)) {
					continue;
				}
				
				Edge edge = roadNetwork.getEdge(currentNode, successor);
				double timeSpent	= averageTravelTime.get(edge.getID()); 
				double temp_g_score = g.get(currentNode) + timeSpent;
				
				if(!openList.contains(successor) || temp_g_score < g.get(successor)) {
					parents.put(successor, currentNode);
					g.put(successor,temp_g_score);
					f.put(successor,temp_g_score+AStar.heuristic(successor, finalNode, maxSpeed));
					if(!openList.contains(successor)) {
						openList.add(successor);
					}
				}
			}
		}
		return null;
	}
	
	private static List<String> reconstructPath(Map<Node, Node> parents,
			Node currentNode, RoadNetwork rn) {
		List<Node> pathNodes = new ArrayList<Node>();
		pathNodes.add(currentNode);
		while(parents.containsKey(currentNode)) {
			currentNode = parents.get(currentNode);
			
			pathNodes.add(currentNode);
		}
		
		List<String> path = new ArrayList<String>(pathNodes.size()-1);
		
		for(int i = 0; i < pathNodes.size()-1;i++) {
			path.add(i,null);
		}
		
		for(int i = 0; i < pathNodes.size()-1;i++) {
			Edge edge = rn.getEdge(pathNodes.get(i+1),pathNodes.get(i));
			int index = pathNodes.size()- 2 - i;
			path.set(index,edge.getID());
		}
		
		return path;
	}

	/**
	 * Gives an estimation of the travel time needed to go from currentNode to finalNode.
	 * The heuristic is the straight line between the two points divided by the maximum driving speed
	 * @param currentNode
	 * @param finalNode
	 * @param roadNetwork
	 * @param maxSpeed
	 * @return an estimation of the travel time needed to go from currentNode to finalNode.
	 */
	public static double heuristic(Node currentNode, Node finalNode,double maxSpeed) {
		double distance = Node.nodeDistance(currentNode, finalNode);
		return distance/maxSpeed; 
	}
	
}
