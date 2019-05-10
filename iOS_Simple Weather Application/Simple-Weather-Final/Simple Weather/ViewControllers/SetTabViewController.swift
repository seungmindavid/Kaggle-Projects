//
//  SetTabViewController.swift
//  Simple Weather
//
//  Created by SEUNG MIN BAEK on 4/17/19.
//  Copyright © 2019 Yoshimoto, Kristy Saori. All rights reserved.
//

//App Name: Simple Weather
//Turned in: April 21
//Members:
//Kristy Yoshimoto kyoshimo@iu.edu
//Michael McQuitty mmcquitt@iu.edu
//Seung Min Baek seubaek@iu.edu
//Kevin Cao kevcao@iu.edu

import UIKit
import MapKit

class SetTabViewController: UITableViewController {
    
    var resultSearchController: UISearchController!
    let locationManager = CLLocationManager()
    
    var myAppDelegate: AppDelegate?
    var myModel: Location?
    var mapView: MKMapView? = nil
    
    var locations: [String] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.myAppDelegate = UIApplication.shared.delegate as! AppDelegate
        self.myModel = self.myAppDelegate?.myModel
        
        self.locations = self.myModel?.name ?? ["No location names"]
        
        locationManager.delegate = self 
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.requestWhenInUseAuthorization()
        locationManager.requestLocation()
        
        let locationSearchTable = storyboard!.instantiateViewController(withIdentifier: "LocationSearchTable") as! LocationSearchTable
        resultSearchController = UISearchController(searchResultsController: locationSearchTable)
        resultSearchController.searchResultsUpdater = locationSearchTable
        
        let searchBar = resultSearchController!.searchBar
        //Make search bar extend the whole screen
        searchBar.sizeToFit()
        //Placeholder text
        searchBar.placeholder = "Search for a place"
        //Put search bar at the top
        navigationItem.titleView = resultSearchController?.searchBar
        
        resultSearchController.hidesNavigationBarDuringPresentation = false
        resultSearchController.dimsBackgroundDuringPresentation = true
        definesPresentationContext = true
        
        //Swipe right to switch to map tab
        let swipeRight = UISwipeGestureRecognizer(target: self, action: #selector(swiped))
        swipeRight.direction = UISwipeGestureRecognizer.Direction.right
        self.view.addGestureRecognizer(swipeRight)
        
    }
    
    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return locations.count
    }
    
    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "setTabViewCell", for: indexPath) as! SetTableViewCell
        
        cell.textLabel?.text = locations[indexPath.item]
        //print(locations[indexPath.item])
        
        return cell
    }
    
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        
        self.myAppDelegate = UIApplication.shared.delegate as! AppDelegate
        self.myModel = self.myAppDelegate?.myModel
        //Use chosen location
        self.myModel?.setCurrentLocIndex(newIndex: indexPath.row)
    }
    
    override func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        
        self.myAppDelegate = UIApplication.shared.delegate as! AppDelegate
        self.myModel = self.myAppDelegate?.myModel
        
        if editingStyle == .delete {
            self.myModel?.removeLocation(index: indexPath.row)
            locations.remove(at: indexPath.row)
            tableView.deleteRows(at: [indexPath], with: .fade)
        }
    }
    
    @objc func swiped(_ gesture: UISwipeGestureRecognizer) {
        if gesture.direction == .right {
            if (self.tabBarController?.selectedIndex)! > 0 {
                self.tabBarController?.selectedIndex -= 1
            }
        }
    }
}

extension SetTabViewController : CLLocationManagerDelegate {
    
    func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        if status == .authorizedWhenInUse {
            //Request again because first time failed
            locationManager.requestLocation()
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("error: \(error)")
    }
}

