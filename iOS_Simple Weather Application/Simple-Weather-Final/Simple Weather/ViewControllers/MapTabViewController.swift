//
//  MapTabViewController.swift
//  Simple Weather
//
//  Created by SEUNG MIN BAEK on 4/16/19.
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
import CoreLocation

class MapTabViewController: UIViewController, CLLocationManagerDelegate {
    
    var myAppDelegate: AppDelegate?
    var myModel: Location?
    var defaults: UserDefaults?

    @IBOutlet weak var myMap: MKMapView!
    @IBOutlet weak var currentLocation: UILabel!
    @IBOutlet weak var currentLongitude: UILabel!
    @IBOutlet weak var currentLatitude: UILabel!
    @IBOutlet weak var currentAddress: UILabel!
    
    private let locationManager = CLLocationManager()
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        self.myAppDelegate = UIApplication.shared.delegate as! AppDelegate
        self.myModel = self.myAppDelegate?.myModel
        self.defaults = self.myAppDelegate?.defaults


        // Bloomington latitude and longitude
        //        let coordinate = CLLocationCoordinate2D(latitude: 39.1637, longitude: -86.524264)

        
        let myLatitude = self.myModel?.latitude[self.myModel?.currentLocIndex ?? 0] ?? 0
        let myLongtitude = self.myModel?.longitude[self.myModel?.currentLocIndex ?? 0] ?? 0

        
        let coordinate = CLLocationCoordinate2D(latitude: myLatitude, longitude: myLongtitude)
        
        self.myMap.region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: myLatitude, longitude: myLongtitude),
                                               span:
            MKCoordinateSpan(latitudeDelta: 1.0, longitudeDelta: 2.0))
        
        //For Annotation
        let pin = MKPointAnnotation()
        pin.title = self.myModel?.name[self.myModel?.currentLocIndex ?? 0] ?? ""
        pin.coordinate = CLLocationCoordinate2D(latitude: myLatitude, longitude: myLongtitude)
        self.myMap.addAnnotation(pin)
        self.myMap.setCenter(coordinate, animated: true)
        // For getting address
        
//        CLGeocoder().reverseGeocodeLocation(CLLocation(latitude: myLatitude, longitude: myLongtitude)) { (placemark, error) in
//            if error != nil{
//                print("ERROR!")
//            }
//            else{
//                if let place = placemark?[0]{
//                    self.currentAddress.text = place.name
//                    let newAddress = place.locality!
//                    if (self.myModel?.getAddress().isEmpty)!{
//                        self.myModel!.addAddress(add: newAddress)
//                    }else if !(self.myModel!.getAddress().contains(newAddress)){
//                        self.myModel!.addAddress(add: newAddress)
//                    }
//                }
//            }
//        }
        
        CLGeocoder().reverseGeocodeLocation(CLLocation(latitude: myLatitude, longitude: myLongtitude)) { (placemark, error) in
            if error != nil {
                print("ERROR!")
                print(error)
            }
            else {
                if let place = placemark?[0]{
                    self.currentAddress.text = "Address: \(place.name ?? "")"
                }
            }
            
        }
        
        self.currentLocation.text = self.myModel?.name[self.myModel?.currentLocIndex ?? 0]
        self.currentLongitude.text = "Longitude: \(myLongtitude)"
        self.currentLatitude.text = "Latitude: \(myLatitude)"
        
        // Set maptype
        let mapType = self.defaults?.string(forKey: "mapType")
        print(mapType!)
        
        if (mapType == "standard") {
            self.myMap.mapType = MKMapType.standard
        } else if (mapType == "satellite") {
            self.myMap.mapType = MKMapType.satellite
        } else if (mapType == "hybrid") {
            self.myMap.mapType = MKMapType.hybrid
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        locationManager.delegate = self
        // for the best accuracy
        
        locationManager.requestWhenInUseAuthorization()
        locationManager.requestAlwaysAuthorization()
        locationManager.startUpdatingLocation()
        // Do any additional setup after loading the view.
        self.myMap.showsUserLocation = true
        
        //Swipe left to switch to set tab
        let swipeLeft = UISwipeGestureRecognizer(target: self, action: #selector(swiped))
        swipeLeft.direction = UISwipeGestureRecognizer.Direction.left
        self.view.addGestureRecognizer(swipeLeft)
        
        //Swipe right to switch to home tab
        let swipeRight = UISwipeGestureRecognizer(target: self, action: #selector(swiped))
        swipeRight.direction = UISwipeGestureRecognizer.Direction.right
        self.view.addGestureRecognizer(swipeRight)
        
//        var userCurrLocCoor = self.myMap.userLocation.coordinate
//        var userCurrLocLat = userCurrLocCoor.latitude
//        var userCurrLocLong = userCurrLocCoor.longitude
//        var userCurrLocName = ""
//        CLGeocoder().reverseGeocodeLocation(CLLocation(latitude: userCurrLocLat, longitude: userCurrLocLong)) {(placemark, error) in
//            if error != nil {
//                print("ERROR!")
//            }
//            else {
//                if let place = placemark?[0]{
//                    userCurrLocName = "\(place.name ?? "")"
//                }
//            }
//
//        }
//        self.myModel?.setCurrentLocation(name: userCurrLocName, lat: userCurrLocLat, long: userCurrLocLong)
//        print("In maptab \(self.myModel?.name)")
//        print(userCurrLocName)
//
//        if let tabVC = self.tabBarController {
//            if let homeVC = tabVC.viewControllers?[0] as? HomeTabBarViewController {
//                homeVC.viewDidLoad()
//            }
//            if let navVC = tabVC.viewControllers?[2] as? UINavigationController {
//                if let setVC = navVC.viewControllers[0] as? SetTabViewController {
//                    setVC.tableView.reloadData()
//                }
//            }
//        }
    }

    @objc func swiped(_ gesture: UISwipeGestureRecognizer) {
        if gesture.direction == .left {
            if (self.tabBarController?.selectedIndex)! < 3 { // set your total tabs here
                self.tabBarController?.selectedIndex += 1
            }
        }
        else if gesture.direction == .right {
            if (self.tabBarController?.selectedIndex)! > 0 {
                self.tabBarController?.selectedIndex -= 1
            }
        }
    }
}
