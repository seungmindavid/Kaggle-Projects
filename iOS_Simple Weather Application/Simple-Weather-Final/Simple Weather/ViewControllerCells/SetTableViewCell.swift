//
//  SetTableViewCell.swift
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

import UIKit

class SetTableViewCell: UITableViewCell {
    @IBOutlet weak var locationTitle: UILabel!
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
