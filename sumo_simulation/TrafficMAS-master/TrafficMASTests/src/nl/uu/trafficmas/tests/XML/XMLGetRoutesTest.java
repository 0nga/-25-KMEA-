package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.ArrayList;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.roadnetwork.RoadNetwork;
import nl.uu.trafficmas.roadnetwork.Route;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLGetRoutesTest {

	@Test
	public void xmlGetRoutes() throws ParserConfigurationException, SAXException, IOException {
		String dir 			= System.getProperty("user.dir")+"/tests/XML/GetRoutes/";
		String nodeXML 		= "NodeTest.xml";
		String edgeXML 		= "EdgeTest.xml";
		String routeXML 	= "RouteTest.xml";
		String nodeXML2 	= "NodeTest2.xml";
		String edgeXML2 	= "EdgeTest2.xml";
		String routeXML2 	= "RouteTest2.xml";
		
		Document nodeDoc 	= DataModelXML.loadDocument(dir, nodeXML);
		Document edgeDoc 	= DataModelXML.loadDocument(dir, edgeXML);
		Document routeDoc	= DataModelXML.loadDocument(dir, routeXML);

		RoadNetwork rn = DataModelXML.instantiateRoadNetwork(nodeDoc, edgeDoc);
		ArrayList<Route> routes = DataModelXML.getRoutes(rn, routeDoc);
		assertEquals(routes.size(),1);
		
		Document nodeDoc2 	= DataModelXML.loadDocument(dir, nodeXML2);
		Document edgeDoc2 	= DataModelXML.loadDocument(dir, edgeXML2);
		Document routeDoc2	= DataModelXML.loadDocument(dir, routeXML2);

		rn = DataModelXML.instantiateRoadNetwork(nodeDoc2, edgeDoc2);
		routes = DataModelXML.getRoutes(rn, routeDoc2);
		assertEquals(routes.size(),2);
		assertEquals(routes.get(1).routeID,"route1");
	}
}
