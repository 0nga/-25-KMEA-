package nl.uu.trafficmas.tests.misc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.junit.Test;

public class LinkedListVSArrayList {

	@Test
	public void test() {
		
		int size = (int) 1e5*4;
		List<Integer> list1 = new ArrayList<Integer>();
		LinkedList<Integer> list2 = new LinkedList<Integer>();
		List<Integer> list3 = new ArrayList<Integer>(size);
		long beginTime;
	
		
		beginTime = System.nanoTime();
	
		for (int i = 0; i < size; i++) {
			list1.add(i);
		}
		//System.out.println("Filling of arraylist took "+(System.nanoTime()-beginTime)/1e6+"ms");
		
		beginTime = System.nanoTime();
		
		for (int i = 0; i < size; i++) {
			list2.add(i);
		}
		//System.out.println("Filling of linkedlist took "+(System.nanoTime()-beginTime)/1e6+"ms");
		
		beginTime = System.nanoTime();
		
		for (int i = 0; i < size; i++) {
			list3.add(i);
		}
		//System.out.println("Filling of prepared arraylist took "+(System.nanoTime()-beginTime)/1e6+"ms");
		
		//sublist
		//System.out.println("Start sublisting");
		beginTime = System.nanoTime();
	
		list1.subList(0, size/2);
		list1.subList(0,size/2).clear();
		//System.out.println("Sublist + clear of arraylist took "+(System.nanoTime()-beginTime)/1e6+"ms");
		
		
	}

}
