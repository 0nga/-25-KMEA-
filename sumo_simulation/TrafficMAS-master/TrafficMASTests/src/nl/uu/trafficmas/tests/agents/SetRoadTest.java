package nl.uu.trafficmas.tests.agents;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.ArrayList;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.agent.Agent;
import nl.uu.trafficmas.agent.NormalAgent;
import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.Road;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class SetRoadTest {

	@Test
	public void setRoad() throws ParserConfigurationException, SAXException, IOException {
		String dir = System.getProperty("user.dir")+"/tests/Agent/SetRoad/";

		String nodXML 	= "takeOverScenario.nod.xml";
		String edgeXML 	= "takeOverScenario.edg.xml";
		String rouXML 	= "takeOverScenario.rou.xml";

		Document nodeDoc 	= DataModelXML.loadDocument(dir, nodXML);
		Document edgeDoc 	= DataModelXML.loadDocument(dir, edgeXML);
		Document rouDoc 	= DataModelXML.loadDocument(dir, rouXML);
		
		RoadNetwork rn 			= DataModelXML.instantiateRoadNetwork(nodeDoc,edgeDoc);
		ArrayList<Route> routes = DataModelXML.getRoutes(rn, rouDoc);		

		Road r 	= rn.getRoadFromID("edge1");
		Agent a = new NormalAgent("Agent1", rn.getNodes()[2], routes.get(0), rn, 6000, 70.0, 70.0);
		// Route consists of 3 roads now.
		assertEquals(3,a.getRoute().length);
		
		Edge secondToLast 	= a.getRoute()[1];
		Edge lastEdge 		= a.getRoute()[2];
		
		// Move one road ahead
		a.setRoad(r);
		
		//Route should be length 2 now
		assertEquals(2,a.getRoute().length);
		
		// And the last two edges should still be in order.
		assertEquals(a.getRoute()[0],secondToLast);
		assertEquals(a.getRoute()[1],lastEdge);
	}
}
