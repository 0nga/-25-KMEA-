package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.roadnetwork.Edge;
import nl.uu.trafficmas.roadnetwork.LaneType;
import nl.uu.trafficmas.roadnetwork.Node;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLExtractEdgesTest {

	@Test
	public void extractEdges() throws ParserConfigurationException, SAXException, IOException {
		
		String dir 		= System.getProperty("user.dir")+"/tests/XML/ExtractEdges/";
		String nodeXML 	= "NodeTest.xml";
		String edgeXML 	= "EdgeTest.xml";
		
		Document nodeDoc = DataModelXML.loadDocument(dir, nodeXML);
		Document edgeDoc = DataModelXML.loadDocument(dir, edgeXML);

		HashMap<String, Node> nodes = DataModelXML.extractNodes(nodeDoc);

		ArrayList<Edge> edges = DataModelXML.extractEdges(edgeDoc, nodes);
		assertNotNull(edges);
		assertEquals(edges.size(),1);
		assertEquals(edges.get(0).getFromNode().name,"A28H0");
		assertEquals(edges.get(0).getToNode().name,"A28H350");
		assertEquals(edges.get(0).getRoad().priority,2);
		assertEquals(edges.get(0).getRoad().getLanes().length,2);

		assertEquals(edges.get(0).getRoad().getLanes()[0].laneType,LaneType.Normal);
		assertEquals(edges.get(0).getRoad().getLanes()[1].laneType,LaneType.Normal);
		double l = Node.nodeDistance(edges.get(0).getFromNode(), edges.get(0).getToNode());
		assertEquals(edges.get(0).getRoad().length, l,0);
		
		
	}

}
