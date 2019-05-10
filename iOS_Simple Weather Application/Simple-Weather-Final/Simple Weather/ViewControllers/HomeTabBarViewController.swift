//
//  HomeTabBarViewController.swift
//  Simple Weather
//
//  Created by Yoshimoto, Kristy Saori on 4/16/19.
//  Copyright © 2019 Yoshimoto, Kristy Saori. All rights reserved.
//

//App Name: Simple Weather
//Turned in: April 21
//Members:
//Kristy Yoshimoto kyoshimo@iu.edu
//Michael McQuitty mmcquitt@iu.edu
//Seung Min Baek seubaek@iu.edu
//Kevin Cao kevcao@iu.edu

import Foundation
import UIKit

class HomeTabBarViewController: UIViewController {
    
    
    var myAppDelegate: AppDelegate?
    var myModel: Location?
    var defaults: UserDefaults?
    var useCelsius: Bool?
    
    let apiKey: String = "d8d52632f4c8544bd2c96004a1ab06d0"
    
    @IBOutlet weak var currentLocation: UILabel!
    @IBOutlet weak var temp: UILabel!
    @IBOutlet weak var desc: UILabel!
    @IBOutlet weak var icon: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        self.myAppDelegate = UIApplication.shared.delegate as! AppDelegate
        self.myModel = self.myAppDelegate?.myModel
        self.defaults = self.myAppDelegate?.defaults
        
        self.useCelsius = self.defaults?.bool(forKey: "tempScale")
        
        self.currentLocation.text = self.myModel?.name[self.myModel?.currentLocIndex ?? 0] ?? ""
        
        var curIndex = self.myModel?.currentLocIndex ?? 0
        
        getLocalWeather(lat: self.myModel?.latitude[curIndex] ?? 0, lon: self.myModel?.longitude[curIndex] ?? 0)
        
        //Swipe left to switch to map tab
        let swipeLeft = UISwipeGestureRecognizer(target: self, action: #selector(swiped))
        swipeLeft.direction = UISwipeGestureRecognizer.Direction.left
        self.view.addGestureRecognizer(swipeLeft)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(true)
        
        self.myAppDelegate = UIApplication.shared.delegate as! AppDelegate
        self.myModel = self.myAppDelegate?.myModel
        
        //UPDATE location text
        self.currentLocation.text = self.myModel?.name[self.myModel?.currentLocIndex ?? 0] ?? ""
        
        var curIndex = self.myModel?.currentLocIndex ?? 0
        
        getLocalWeather(lat: self.myModel?.latitude[curIndex] ?? 0, lon: self.myModel?.longitude[curIndex] ?? 0)
    }
    
    func getLocalWeather(lat: Double, lon: Double) {
        let session = URLSession.shared
        if let weatherURL = URL(string: "https://api.openweathermap.org/data/2.5/weather?lat=\(lat)&lon=\(lon)&APPID=\(apiKey)") {
            
            let dataTask = session.dataTask(with: weatherURL) {
                (data: Data?, response: URLResponse?, error: Error?) in
                if let error = error {
                    print(error)
                } else {
                    if let data = data {
                        let dString = String(data: data, encoding: String.Encoding.utf8)
                        //print("All the weather data: \n\(dString!)")
                        if let jsonObj = try? JSONSerialization.jsonObject(with: data, options: .allowFragments) as? NSDictionary {
                            
                            if let mainDictionary = jsonObj!.value(forKey: "main") as? NSDictionary {
                                
                                if let temperature = mainDictionary.value(forKey: "temp") as! Double? {
                                    DispatchQueue.main.async {
                                        var temp = (temperature - 273.15)
                                        
                                        if (!self.useCelsius!) {
                                            temp = temp * (9 / 5) + 32
                                        }
                                        //print(f)
                                        self.temp.text = String(format: "Today: %.2f", temp)
                                        
                                        if (self.useCelsius!) {
                                            self.temp.text? += "C"
                                        } else {
                                            self.temp.text? += "F"
                                        }
                                    }
                                }
                                
                            } else {
                                print("not found 1")
                            }
                            
                            if let main2Dictionary = jsonObj!.value(forKey: "weather") as? [Any] {
                                let weatherDict : NSDictionary = main2Dictionary[0] as! NSDictionary
                                if let description = weatherDict.value(forKey: "description") {
                                    DispatchQueue.main.async {
                                        let descr = description
                                        self.desc.text = "Description:  \(descr)"
                                    }
                                }
                                //Changes icon depending on main description
                                if let mainDesc = weatherDict.value(forKey: "main") {
                                    DispatchQueue.main.async {
                                        let mainD = mainDesc
                                        self.changeImage(desc: mainD)
                                    }
                                }
                                
                            } else {
                                print("not found 2")
                            }
                            
                        } else {
                            print("cant convert json data")
                        }
                    } else {
                        print("Error: did not recieve data")
                    }
                }
            }
            dataTask.resume()
        } else {
            print("error unwrapping url")
        }
    }
    
    func changeImage(desc: Any) {
        let s = String(describing: desc)
        //print(s)
        switch s {
        case "Clear":
            self.icon.image = UIImage(named: "Sun")
            break
        case "Thunderstorm":
            self.icon.image = UIImage(named: "Lightning")
            break
        case "Drizzle":
            self.icon.image = UIImage(named: "Rain")
            break
        case "Rain":
            self.icon.image = UIImage(named: "Rain")
            break
        case "Snow":
            self.icon.image = UIImage(named: "Snow")
            break
        case "Clouds":
            self.icon.image = UIImage(named: "Cloud")
            break
        case "Tornado":
            self.icon.image = UIImage(named: "Tornado")
            break
        default:
            self.icon.image = UIImage(named: "Wind")
            break
        }
    }
    
    @objc func swiped(_ gesture: UISwipeGestureRecognizer) {
        if gesture.direction == .left {
            if (self.tabBarController?.selectedIndex)! < 3 { // set your total tabs here
                self.tabBarController?.selectedIndex += 1
            }
        }
    }
}
