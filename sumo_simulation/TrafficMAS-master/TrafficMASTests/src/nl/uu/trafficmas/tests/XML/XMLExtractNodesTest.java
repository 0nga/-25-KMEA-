package nl.uu.trafficmas.tests.XML;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.HashMap;

import javax.xml.parsers.ParserConfigurationException;

import nl.uu.trafficmas.datamodel.DataModelXML;
import nl.uu.trafficmas.roadnetwork.Node;

import org.junit.Test;
import org.w3c.dom.Document;
import org.xml.sax.SAXException;

public class XMLExtractNodesTest {

	@Test
	public void extractNodes() throws ParserConfigurationException, SAXException, IOException {
//		<node id="A28H0" x="-350.0" y="0.0" />
//	    <node id="A28H350" x="0.0" y="0.0" /> 
		String dir 		= System.getProperty("user.dir")+"/tests/XML/ExtractNodes/";
		String nodeXML 	= "NodeTest.xml";
		
		Document nodeDoc = DataModelXML.loadDocument(dir, nodeXML);
		HashMap<String, Node> nodes = DataModelXML.extractNodes(nodeDoc);

		assertEquals(nodes.size(),2); 
		assertEquals(nodes.get("A28H0"),new Node("A28H0",-350,0));
		assertEquals(nodes.get("A28H350"),new Node("A28H350",0,0));
	}
}
