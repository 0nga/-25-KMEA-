package nl.uu.trafficmas.tests.roadNetwork;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Node;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;

import org.junit.Test;

public class AddEdgesTest {

	@Test
	public void addEdges() {
		RoadNetwork rn = new RoadNetwork();
		
		Node n1 = new Node("n1",0,0);
		Node n2 = new Node("n2",0,0);
		Road r1 = new Road("r1",5.0,null,1);
		Road r2 = new Road("r2",6.0,null,1);

		Edge edge1 = new Edge(n1,n1,r1);
		Edge edge2 = new Edge(n1,n2,r2);
		Edge edge3 = new Edge(n2, n2, r2);
		
		ArrayList<Edge> edges = new ArrayList<Edge>();
		edges.add(edge1);
		edges.add(edge2);
		edges.add(edge3);

		rn.addEdges(edges);
		Set<Edge> edgeSet = new HashSet<Edge>();
		for(Edge e: rn.getEdges()) {
			edgeSet.add(e);
		}
		
		assertEquals(rn.getEdges().length, 3);
		assertEquals(true, edgeSet.contains(edge1));
		assertEquals(true, edgeSet.contains(edge2));
		assertEquals(true, edgeSet.contains(edge3));
	
	}
}
